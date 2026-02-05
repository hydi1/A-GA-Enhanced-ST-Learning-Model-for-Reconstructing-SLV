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
        'model_id': 456,
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
     'input_size': 4 * 33 * 5 * 9,
        'root_path': r'D:\sea level variability\DATA_neao',
        "data_path": 'Anomalies_2004-2022_filtered.npy',
        "target_path": r"D:\sea level variability\DATA_neao\4processed_HIERRO_nomiss.xlsx" ,
        'target': "OT",
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention':False,
        "lradj": "type1",

        'checkpoints': r'D:\sea level variability\code_neao\不同回溯窗口\SOFTS_TS -12\SOFTS-main\checkpoints',
        "save_model":True,
        'device_ids':[0],
        'scale': True,

    }

class CustomDataset(Dataset):
    def __init__(self):
        self.__loda_data__()

    def __loda_data__(self):

        self.scaler_x = joblib.load(
            r"D:\sea level variability\code_neao\不同回溯窗口\SOFTS_TS -12\SOFTS-main\scaler_x_time.pkl")
        self.scaler_y = joblib.load(
            r"D:\sea level variability\code_neao\不同回溯窗口\SOFTS_TS -12\SOFTS-main\scaler_y_time.pkl")

        file_path = r"D:\sea level variability\DATA_neao\Anomalies_1993-2023.npy"
        all_x_data = np.load(file_path)
        all_x_data_2d = all_x_data.reshape(-1, args['input_size'])
        self.data_x = self.scaler_x.transform(all_x_data_2d)

        y_path = r"D:\sea level variability\DATA_neao\4processed_HIERRO_372.xlsx"
        df_y = pd.read_excel(y_path)
        y_raw = df_y.iloc[:, 1].values.reshape(-1, 1)
        self.y_data = y_raw

        y_data_filled = np.where(np.isnan(y_raw), np.nanmean(y_raw), y_raw)
        self.y_data_normalized = self.scaler_y.transform(y_data_filled)

        time_path = r"D:\goole\GOPRdata\time_data.xlsx"
        time_df = pd.read_excel(time_path)
        df_stamp = pd.to_datetime(time_df.iloc[:, 0].values)
        self.data_stamp = time_features(pd.DatetimeIndex(df_stamp), freq=args['freq'])
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
        seq_x_batch, seq_y_batch, batch_x_mark, batch_y_mark = zip(*batch)
        return torch.stack(seq_x_batch, 0), torch.stack(seq_y_batch, 0), \
            torch.stack(batch_x_mark, 0), torch.stack(batch_y_mark, 0)

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

ckpt_path = r"D:\sea level variability\code_neao\不同回溯窗口\SOFTS_TS -12\SOFTS-main\checkpoints\TSdepth51993-2023_456_SOFTS_ssta_MS_0.0005_12_12_12_64_2_1_256\checkpoint.pth"
state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

test_data, test_loader = data_provider()

total_steps = 372
full_sequence = np.zeros((total_steps, 1))
full_y_true_sum = np.zeros((total_steps, 1))
counts = np.zeros((total_steps, 1))

with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x_mark, batch_y_mark = batch_x_mark.to(device), batch_y_mark.to(device)

        output = model(batch_x, batch_x_mark, batch_y, batch_y_mark)

        outputs_flat = output.cpu().numpy().reshape(-1, 1)
        batch_y_flat = batch_y.cpu().numpy().reshape(-1, 1)

        outputs_unnorm = test_data.scaler_y.inverse_transform(outputs_flat)
        batch_y_unnorm = test_data.scaler_y.inverse_transform(batch_y_flat)

        outputs_res = outputs_unnorm.reshape(output.shape)
        batch_y_res = batch_y_unnorm.reshape(batch_y.shape)

        batch_size = outputs_res.shape[0]
        start_idx = i * args['batch_size']

        for b in range(batch_size):
            p_start = start_idx + b
            p_end = p_start + args['pred_len']
            v_end = min(p_end, total_steps)
            s_len = v_end - p_start

            full_sequence[p_start:v_end] += outputs_res[b, :s_len, 0].reshape(-1, 1)
            full_y_true_sum[p_start:v_end] += batch_y_res[b, :s_len, 0].reshape(-1, 1)
            counts[p_start:v_end] += 1

full_sequence /= np.maximum(counts, 1)
full_y_true_avg = full_y_true_sum / np.maximum(counts, 1)

full_y_true_raw = test_data.y_data
df = pd.DataFrame({
    'True_Values': full_y_true_raw.flatten(),
    'Predicted_Values': full_sequence.flatten()
})
df.to_excel('SOFTS_reconstructed_372.xlsx', index=False)

mask = ~np.isnan(full_y_true_raw) & ~np.isnan(full_sequence)
rmse = np.sqrt(np.mean((full_y_true_raw[mask] - full_sequence[mask]) ** 2))
print(f"SOFTS RMSE (non-NaN): {rmse:.4f}")
print("Reconstructed shape:", full_sequence.shape)
