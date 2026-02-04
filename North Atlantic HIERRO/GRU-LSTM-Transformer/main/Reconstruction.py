import pandas as pd
import numpy as np
import torch
import joblib
from torch.utils.data import DataLoader, Dataset
from models import Model  # 核心：利用 __init__.py 的自动分流逻辑
from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler

# ================= 1. 配置参数 =================
args = {
    'task_name': 'depth1993-2023',
    'model_id': 44,
    'model': 'LSTM',  # 确保这里与权重文件匹配，自动切换到 LSTM.py
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
    'hidden_size': 64,
    'output_size': 1,
    'num_layers': 3,
    'root_path': r'D:\sea level variability\DATA_neao',
    "data_path": 'Anomalies_2004-2022_filtered.npy',
    "target_path": r"D:\sea level variability\DATA_neao\4processed_HIERRO_nomiss.xlsx",
    'target': "OT",
    'seasonal_patterns': 'Monthly',
    'num_workers': 4,
    'use_amp': False,
    'output_attention': False,
    "lradj": "type1",
    'checkpoints': r'D:\sea level variability\neaocode\不同回溯窗口\GRU - 12\SOFTS-main\checkpoints',
    "save_model": True,
    'device_ids': [0],
    'scale': True,
}


# ================= 2. 数据集类定义 =================
class CustomDataset(Dataset):
    def __init__(self):
        self.__loda_data__()

    def __loda_data__(self):
        # 加载训练时保存的 scaler (必须用 transform 保证分布一致)
        self.scaler_x = joblib.load(
            r"D:\sea level variability\code_neao\不同回溯窗口\GRU - 12\SOFTS-main\scaler_x_time.pkl")
        self.scaler_y = joblib.load(
            r"D:\sea level variability\code_neao\不同回溯窗口\GRU - 12\SOFTS-main\scaler_y_time.pkl")

        # 加载 X 数据
        file_path = r"D:\sea level variability\DATA_neao\Anomalies_1993-2023.npy"
        all_x_data = np.load(file_path)
        all_x_data_2d = all_x_data.reshape(-1, args['input_size'])
        self.data_x = self.scaler_x.transform(all_x_data_2d)

        # 加载 Y 数据
        y_path = r"D:\sea level variability\DATA_neao\4processed_HIERRO_372.xlsx"
        df_y = pd.read_excel(y_path)
        self.y_data = df_y.iloc[:, 1].values.reshape(-1, 1)  # 原始包含 NaN 的数据

        # 数据清洗：前向填充 + 均值填充
        y_series = df_y.iloc[:, 1]
        y_filled = y_series.fillna(method='ffill')
        if y_filled.isna().sum() > 0:
            y_filled = y_filled.fillna(y_filled.mean())

        y_data_clean = y_filled.values.reshape(-1, 1)
        self.y_data_scaled = self.scaler_y.transform(y_data_clean)

        # 时间特征加载
        time_data = df_y.iloc[:, 0].values
        df_stamp = pd.to_datetime(time_data)
        dates = pd.DatetimeIndex(df_stamp)
        self.data_stamp = time_features(dates, freq='D')
        self.data_stamp = self.data_stamp.transpose(1, 0)

    def __len__(self):
        return len(self.data_x) - args['seq_len'] + 1

    def __getitem__(self, index):
        seq_x = torch.tensor(self.data_x[index:index + args['seq_len']], dtype=torch.float32)
        seq_y = torch.tensor(self.y_data_scaled[index:index + args['seq_len']], dtype=torch.float32)
        seq_x_mark = torch.tensor(self.data_stamp[index:index + args['seq_len']], dtype=torch.float32)
        seq_y_mark = torch.tensor(self.data_stamp[index:index + args['seq_len']], dtype=torch.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark


# ================= 3. 数据提供者函数 =================
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


# ================= 4. 模型加载与初始化 =================
# 此处 Model 会自动根据 args['model'] 的值去 models/__init__.py 找对应的类
model = Model(args)

checkpoint_path = r"D:\sea level variability\neaocode\不同回溯窗口\GRU - 12\SOFTS-main\checkpoints\depth1993-2023_44_LSTM_ssta_MS_0.0005_12_12_12_64_2_1_256\checkpoint.pth"
state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# 打印调试，确保参数名一致
print("模型层级结构参数名:", [n for n, p in model.named_parameters()][:5])
model.load_state_dict(state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# ================= 5. 执行推理与窗口叠加 =================
test_data, test_loader = data_provider()

total_steps = 372
full_sequence = np.zeros((total_steps, 1))
full_y_true = np.zeros((total_steps, 1))
counts = np.zeros((total_steps, 1))

with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x_mark, batch_y_mark = batch_x_mark.to(device), batch_y_mark.to(device)

        # 这里的参数根据你的 LSTM/GRU forward 定义传入
        # 一般顺序为: (x_enc, x_mark_enc, x_dec, x_mark_dec)
        output = model(batch_x, batch_x_mark, batch_y, batch_y_mark)

        # 转为 numpy 并反归一化
        outputs_flat = output.cpu().numpy().reshape(-1, 1)
        batch_y_flat = batch_y.cpu().numpy().reshape(-1, 1)

        outputs_unnorm = test_data.scaler_y.inverse_transform(outputs_flat)
        batch_y_unnorm = test_data.scaler_y.inverse_transform(batch_y_flat)

        outputs_original = outputs_unnorm.reshape(output.shape)
        batch_y_original = batch_y_unnorm.reshape(batch_y.shape)

        batch_size = outputs_original.shape[0]
        start_idx = i * args['batch_size']

        for b in range(batch_size):
            for s in range(args['seq_len']):
                t = start_idx + b + s
                if t < total_steps:
                    full_sequence[t] += outputs_original[b, s, 0]
                    full_y_true[t] += batch_y_original[b, s, 0]
                    counts[t] += 1

# 计算平均值
full_sequence /= np.maximum(counts, 1)
full_y_true /= np.maximum(counts, 1)

# 保存结果
full_y_true_raw = test_data.y_data  # 原始含 NaN 值
df = pd.DataFrame({
    'True_Values': full_y_true_raw.flatten(),
    'Predicted_Values': full_sequence.flatten()
})
df.to_excel('reconstructed_372_timesteps_finalTransformer.xlsx', index=False)

# 计算 RMSE
mask = ~np.isnan(full_y_true_raw) & ~np.isnan(full_sequence)
rmse = np.sqrt(np.mean((full_y_true_raw[mask] - full_sequence[mask]) ** 2))
print(f"Final RMSE (non-NaN points): {rmse:.4f}")