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
    'task_name': 'TSUV1',
    'model_id': 'train',
    'model': 'SOFTS',
    'data': 'ssta',
    'features': 'MS',
    'learning_rate': 0.0001,
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
    'input_size': 1056,
    'hidden_size': 64,
    'output_size': 1,
    'num_layers': 3,
    'root_path': r'../../Data',
    "data_path": 'TSuv_data_368.npy',
    "target_path": r"../../Data/Y_nonan-1993_2023.xlsx",
    'target': "OT",
    'seasonal_patterns': 'Monthly',
    'num_workers': 4,
    'use_amp': False,
    'output_attention': False,
    "lradj": "type1",

    'checkpoints': r'../../checkpoints',
    "save_model": True,
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
class CustomDataset(Dataset):
    def __init__(self):
        self.__loda_data__()

    def __loda_data__(self):

        self.scaler_x = joblib.load(r"scaler_x_time.pkl")
        self.scaler_y = joblib.load(r"scaler_y_time.pkl")

        file_path = r"../../Data/combined_result.npy"
        all_x_data = np.load(file_path)
        all_x_data_2d = all_x_data.reshape(-1, 4*29*6*11)
        self.data_x = self.scaler_x.transform(all_x_data_2d)

        y_path = r"../../Data/Y_nonan-1993_2023.xlsx"
        y_data = pd.read_excel(y_path).iloc[:, 1].values.reshape(-1, 1)
        self.y_data = y_data

        y_data_filled = np.where(np.isnan(y_data), np.nanmean(y_data), y_data)
        self.y_data_normalized = self.scaler_y.transform(y_data_filled)

        time_path = r"../../Data/Y_nonan-1993_2023.xlsx"
        time_data = pd.read_excel(time_path).iloc[:, 0].values
        df_stamp = pd.to_datetime(time_data)
        dates = pd.DatetimeIndex(df_stamp)
        self.data_stamp = time_features(dates, freq='D')
        self.data_stamp = self.data_stamp.transpose(1, 0)

    def __len__(self):
        return len(self.data_x) - args['seq_len'] + 1

    def __getitem__(self, index):
        if index >= len(self.data_stamp):
            raise IndexError(f"Index {index} out of bounds for axis 0 with size {len(self.data_stamp)}")

        seq_x = torch.tensor(self.data_x[index:index + args['seq_len']], dtype=torch.float32)
        seq_y = torch.tensor(self.y_data_normalized[index:index + args['seq_len']], dtype=torch.float32)
        seq_x_mark = torch.tensor(self.data_stamp[index:index + args['seq_len']], dtype=torch.float32)
        seq_y_mark = torch.tensor(self.data_stamp[index:index + args['seq_len']], dtype=torch.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

def data_provider():

    def collate_fn(batch):
        try:
            seq_x_batch, seq_y_batch, batch_x_mark, batch_y_mark = zip(*batch)

            seq_x_batch = torch.stack(seq_x_batch, dim=0)
            seq_y_batch = torch.stack(seq_y_batch, dim=0)

            batch_x_mark = [
                torch.tensor(np.array(item), dtype=torch.float32) if not isinstance(item, torch.Tensor) else item for
                item in batch_x_mark]

            batch_x_mark = torch.stack(batch_x_mark, dim=0)

            batch_y_mark = [
                torch.tensor(np.array(item), dtype=torch.float32) if not isinstance(item, torch.Tensor) else item for
                item in batch_y_mark]
            batch_y_mark = torch.stack(batch_y_mark, dim=0)
            return seq_x_batch, seq_y_batch, batch_x_mark, batch_y_mark
        except Exception as e:
            print("Batch processing error:", e)
            raise

    data_set=CustomDataset()
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=0,
        drop_last=False, collate_fn=collate_fn)

    return data_set, data_loader

test_data, test_loader= data_provider()

model.eval()

full_sequence = np.zeros((372, 1))
full_y_true = np.zeros((372, 1))
counts = np.zeros((372, 1))

all_outputs = []
full_sequence = np.zeros((372, 1))
counts = np.zeros((372, 1))
with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        output = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        print(f"Batch {i+1} model_outputs shape: {output.shape}")

        outputs_cpu = output.cpu().numpy()
        batch_y_cpu = batch_y.cpu().numpy()

        outputs_flat = outputs_cpu.reshape(-1, 1)
        batch_y_flat = batch_y_cpu.reshape(-1, 1)
        outputs_fgyh = test_data.scaler_y.inverse_transform(outputs_flat)
        batch_y_fgyh = test_data.scaler_y.inverse_transform(batch_y_flat)

        outputs_original = outputs_fgyh.reshape(outputs_cpu.shape)
        batch_y_original = batch_y_fgyh.reshape(batch_y_cpu.shape)

        batch_size = outputs_original.shape[0]
        start_idx = i * args['batch_size']
        pred_len = args['pred_len']

        for j in range(batch_size):
            pred_start = start_idx + j
            pred_end = pred_start + pred_len
            valid_end = min(pred_end, 372)
            slice_len = valid_end - pred_start

            full_sequence[pred_start:valid_end] += outputs_original[j, :slice_len, 0].reshape(-1, 1)
            full_y_true[pred_start:valid_end] += batch_y_original[j, :slice_len, 0].reshape(-1, 1)
            counts[pred_start:valid_end] += 1

    full_sequence = full_sequence / np.maximum(counts, 1)

    full_y_true_raw = test_data.y_data
    full_y_true = full_y_true / np.maximum(counts, 1)

    df = pd.DataFrame({
        'True_Values': full_y_true_raw.flatten(),

        'Predicted_Values': full_sequence.flatten()
    })
    df.to_excel('reconstructed_372_timesteps_with_true2.xlsx', index=False)

    mask = ~np.isnan(full_y_true_raw) & ~np.isnan(full_sequence)
    rmse = np.sqrt(np.mean((full_y_true_raw[mask] - full_sequence[mask]) ** 2))
    print(f"RMSE (non-NaN points): {rmse:.4f}")
    print("Reconstructed shape:", full_sequence.shape)
    print("True values shape:", full_y_true_raw.shape)
