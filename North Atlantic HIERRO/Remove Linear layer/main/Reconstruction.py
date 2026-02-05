import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models.GAconvgru import Model
from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler
import joblib
args = {
        'task_name': 'GAConvgru_610',
        'model_id': 1888,
        'model': 'GAconvgru',
        'data': 'ssta',
        'features': 'MS',
        'learning_rate': 0.0005,
        'seq_len': 12,
        'label_len': 12,
        'pred_len': 12,
        'd_model': 64,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 256,
        'factor': 1,
        'embed': 'timeF',
        'distil': True,
        'dropout': 0.0,
        'activation': 'gelu',
        'use_gpu': True,
        'train_epochs': 128,
        'batch_size': 16,
        'patience': 128,
        'use_norm': False,
        'd_core': 512,
        'freq': 'D',
        'input_size': 720,
        'hidden_size': 64,
        'output_size': 1,
        'num_layers': 3,
        'root_path': r'../../Data',
        'data_path': 'Anomalies_2004-2022_filtered_reordered.npy',
        'target_path': r"../../Data/4processed_HIERRO_nomiss.xlsx",
        'target': 'OT',
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        'lradj': 'type1',
        'checkpoints': r'../../checkpoints',
        'save_model': True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }

model = Model(args)

state_dict = torch.load(r"../../checkpoints/checkpoint.pth",weights_only=True)

state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
SEQ_LEN = args['seq_len']

SCALER_X_PATH = r"scaler_x_time.pkl"
SCALER_Y_PATH = r"scaler_y_time.pkl"

X_NPY_PATH  = r"../../Data/Anomalies_2004-2022_filtered.npy"
Y_XLSX_PATH = r"../../Data/4processed_HIERRO_nomiss.xlsx"

class CustomDataset(Dataset):
    def __init__(self):
        self.__load_data__()

    def __load_data__(self):
        self.scaler_x = joblib.load(SCALER_X_PATH)
        self.scaler_y = joblib.load(SCALER_Y_PATH)

        all_x = np.load(X_NPY_PATH)
        self.T = all_x.shape[0]
        all_x_2d = all_x.reshape(self.T, -1)
        self.data_x = self.scaler_x.transform(all_x_2d)

        df_y = pd.read_excel(Y_XLSX_PATH)
        y_raw = df_y.iloc[:, 1].values.reshape(-1, 1)
        self.y_data = y_raw

        y_filled = y_raw.copy()
        mean_val = np.nanmean(y_filled)
        y_filled[np.isnan(y_filled)] = mean_val
        self.y_norm = self.scaler_y.transform(y_filled)

        time_data = df_y.iloc[:, 0].values
        dates = pd.DatetimeIndex(pd.to_datetime(time_data))
        self.data_stamp = time_features(dates, freq='D').transpose(1, 0)

    def __len__(self):
        return self.T - SEQ_LEN + 1

    def __getitem__(self, idx):
        seq_x = torch.tensor(self.data_x[idx:idx + SEQ_LEN], dtype=torch.float32)
        seq_y = torch.tensor(self.y_norm[idx:idx + SEQ_LEN], dtype=torch.float32)
        seq_x_mark = torch.tensor(self.data_stamp[idx:idx + SEQ_LEN], dtype=torch.float32)
        seq_y_mark = torch.tensor(self.data_stamp[idx:idx + SEQ_LEN], dtype=torch.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

def data_provider():
    def collate_fn(batch):
        seq_x, seq_y, x_mark, y_mark = zip(*batch)
        return (torch.stack(seq_x, 0),
                torch.stack(seq_y, 0),
                torch.stack([torch.as_tensor(np.array(i), dtype=torch.float32) for i in x_mark], 0),
                torch.stack([torch.as_tensor(np.array(i), dtype=torch.float32) for i in y_mark], 0))

    ds = CustomDataset()
    dl = DataLoader(ds, batch_size=args['batch_size'], shuffle=False,
                    num_workers=0, drop_last=False, collate_fn=collate_fn)
    return ds, dl

test_data, test_loader = data_provider()

model.eval()
device = next(model.parameters()).device

FULL_LENGTH = test_data.T

pred_full = np.zeros((FULL_LENGTH, 1), dtype=np.float64)
counts = np.zeros(FULL_LENGTH, dtype=np.float64)

sample_global_idx = 0

with torch.no_grad():
    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x_mark = batch_x_mark.to(device)
        batch_y_mark = batch_y_mark.to(device)

        dec_inp = torch.zeros_like(batch_y).float().to(device)

        output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if args['features'] == 'MS' else 0
        output = output[:, :, f_dim:]

        out_np = output.detach().cpu().numpy()
        B, L, _ = out_np.shape

        for i in range(B):
            start_idx = sample_global_idx
            end_idx = start_idx + L

            actual_end = min(end_idx, FULL_LENGTH)
            length_to_add = actual_end - start_idx

            if length_to_add > 0:
                pred_full[start_idx:actual_end, 0] += out_np[i, :length_to_add, 0]
                counts[start_idx:actual_end] += 1

            sample_global_idx += 1

counts[counts == 0] = 1
pred_full[:, 0] /= counts

pred_denorm = test_data.scaler_y.inverse_transform(pred_full)

true_raw = test_data.y_data

df = pd.DataFrame({
    "time_idx": np.arange(FULL_LENGTH),
    "true": true_raw.flatten(),
    "pred": pred_denorm.flatten(),

    "counts": counts
})
df.to_excel("NEAO_reconstructed_372_keep_nan_true.xlsx", index=False)

mask = ~np.isnan(true_raw.flatten()) & ~np.isnan(pred_denorm.flatten())
rmse = np.sqrt(np.mean((true_raw.flatten()[mask] - pred_denorm.flatten()[mask]) ** 2))
mae  = np.mean(np.abs(true_raw.flatten()[mask] - pred_denorm.flatten()[mask]))
print(f"RMSE (non-NaN): {rmse:.4f}")
print(f"MAE  (non-NaN): {mae:.4f}")
