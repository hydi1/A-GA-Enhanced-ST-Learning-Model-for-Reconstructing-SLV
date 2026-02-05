import pandas as pd
import numpy as np
import pickle
import joblib
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
from torch.utils.data import Dataset
from models.SOFTS import Model
import joblib
from utils.timefeatures import time_features
args = {
        'task_name': 'TSdepth51993-2023',
        'model_id': 'train',
        'model': 'SOFTS',
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

        "use_norm": False,
        'd_core': 512,
        'freq': 'D',
        'root_path': r'../../Data',
        'data_path': "anomaly_1993_2021_depth15_filtered.npy",
        'target_path': r"../../Data/processed_1589.xlsx",
        'target': "OT",
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention':False,
        "lradj": "type1",

        'checkpoints': r'../../checkpoints',
        "save_model":True,
        'device_ids':[0],
        'scale': True,

        }

class CustomDataset(Dataset):
    def __init__(self):
        self.__loda_data__()

    def __loda_data__(self):

        file_path = r"../../Data/anomaly_1993_2018_depth15_filtered.npy"
        all_x_data = np.load(file_path)

        all_x_data_2d = all_x_data.reshape(-1, 4 * 15 * 11 * 15)

        if all_x_data_2d.dtype != np.float64:
            print("Warning: Input Data type is {}, converted to np.float64".format(all_x_data_2d.dtype))
            all_x_data_2d = all_x_data_2d.astype(np.float64)

        global_mean = np.nanmean(all_x_data_2d[np.isfinite(all_x_data_2d)])

        if np.any(np.isnan(all_x_data_2d)):
            print("Warning: Input data contains NaN, filled with mean")
            all_x_data_2d = np.nan_to_num(all_x_data_2d, nan=global_mean)

        if np.any(np.isinf(all_x_data_2d)):
            print("Warning: Input data contains Inf, filled with mean")
            all_x_data_2d = np.nan_to_num(all_x_data_2d, posinf=global_mean, neginf=global_mean)

        if np.any(np.abs(all_x_data_2d) > 1e8):
            print("Warning: Input data contains extreme values, performing robust normalization")

            mean_tmp = np.nanmean(all_x_data_2d, axis=0)
            std_tmp = np.nanstd(all_x_data_2d, axis=0)
            all_x_data_2d = (all_x_data_2d - mean_tmp) / (std_tmp + 1e-8)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        all_x_data_scaled = scaler.fit_transform(all_x_data_2d)

        if np.any(np.isnan(all_x_data_scaled)) or np.any(np.isinf(all_x_data_scaled)):
            print("Error: Data still contains NaN or Inf after Normalization, trying to force zeroing")
            all_x_data_scaled = np.nan_to_num(all_x_data_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        scaler_save_path = r'../../Data/new_scaler_anomalies_2d.pkl'
        joblib.dump(scaler, scaler_save_path)
        self.scaler_x = joblib.load(scaler_save_path)
        self.data_x = all_x_data_scaled

        y_path = r"../../Data/processed_1589.xlsx"
        df_y = pd.read_excel(y_path)
        y_data = df_y.iloc[:, 1].values.reshape(-1, 1)
        self.y_data = y_data

        y_mean = np.nanmean(y_data)
        y_data_filled = np.where(np.isnan(y_data), y_mean, y_data)
        self.scaler_y = joblib.load(
            r"scaler_y_time.pkl")
        self.y_data_normalized = self.scaler_y.transform(y_data_filled)

        time_data = df_y.iloc[:, 0].values
        df_stamp = pd.to_datetime(time_data)
        self.data_stamp = time_features(pd.DatetimeIndex(df_stamp), freq='D')
        self.data_stamp = self.data_stamp.transpose(1, 0)

    def __len__(self):
        return len(self.data_x) - args['seq_len'] + 1

    def __getitem__(self, index):
        seq_x = torch.tensor(self.data_x[index:index + args['seq_len']], dtype=torch.float32)
        seq_y = torch.tensor(self.y_data_normalized[index:index + args['seq_len']], dtype=torch.float32)
        seq_x_mark = torch.tensor(self.data_stamp[index:index + args['seq_len']], dtype=torch.float32)
        seq_y_mark = torch.tensor(self.data_stamp[index:index + args['seq_len']], dtype=torch.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

def data_provider():
    def collate_fn(batch):
        seq_x, seq_y, x_mark, y_mark = zip(*batch)
        return torch.stack(seq_x, 0), torch.stack(seq_y, 0), \
            torch.stack(x_mark, 0), torch.stack(y_mark, 0)

    data_set = CustomDataset()
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn)
    return data_set, data_loader

model = Model(args)

checkpoint_path = r"../../checkpoints/checkpoint.pth"
state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

test_data, test_loader = data_provider()

total_steps = 372
full_sequence = np.zeros((total_steps, 1))
counts = np.zeros((total_steps, 1))

model.eval()
with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        output = model(batch_x, batch_x_mark, batch_y, batch_y_mark)

        outputs_cpu = output.detach().cpu().numpy()

        outputs_flat = outputs_cpu.reshape(-1, 1)
        outputs_unnorm = test_data.scaler_y.inverse_transform(outputs_flat)
        outputs_original = outputs_unnorm.reshape(outputs_cpu.shape)

        current_batch_size = outputs_original.shape[0]
        start_idx = i * args['batch_size']

        for j in range(current_batch_size):
            p_start = start_idx + j
            p_end = p_start + args['pred_len']

            v_end = min(p_end, total_steps)
            s_len = v_end - p_start

            if s_len > 0:
                full_sequence[p_start:v_end] += outputs_original[j, :s_len, 0].reshape(-1, 1)
                counts[p_start:v_end] += 1

final_output = full_sequence / np.where(counts == 0, 1, counts)

print(f"Coverage count of last 10 points: {counts[-10:].flatten()}")
print(f"Do the last 10 prediction values contain NaN: {np.isnan(final_output[-10:]).any()}")

full_y_true_raw = test_data.y_data

df_res = pd.DataFrame({
    'True_Values': full_y_true_raw.flatten(),
    'Predicted_Values': final_output.flatten()
})

df_res.to_excel('reconstructed_372_timesteps_with_true2SOFTS.xlsx', index=False)
print("Saved successfully! File contains 372 lines of data.")
