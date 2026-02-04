import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models.GAconvgru import Model  # 根据文件和类的实际名称调整路径
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
        'root_path': r'D:\sea level variability\DATA_neao',
        'data_path': 'Anomalies_2004-2022_filtered_reordered.npy',
        'target_path': r"D:\sea level variability\DATA_neao\4processed_HIERRO_nomiss.xlsx",
        'target': 'OT',
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        'lradj': 'type1',
        'checkpoints': r'D:\sea level variability\code_neao\convgru_TSVU - 反归一化\SOFTS-main\checkpoints',
        'save_model': True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }



# 假设模型初始化
model = Model(args)

# 加载模型参数
state_dict = torch.load(r"D:\sea level variability\code_neao\convgru_TSVU - 反归一化\SOFTS-main\checkpoints\GAConvgru_610_seed1888_GAconvgru_ssta_MS_0.0005_12_12_12_64_2_1_256\checkpoint.pth",weights_only=True)

# 移除 'module.' 前缀
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
# print("Before loading, first parameter sample:", next(iter(model.parameters()))[0])  # 打印初始参数
# 加载更新后的 state_dict
model.load_state_dict(state_dict)
# print("After loading, first parameter sample:", next(iter(model.parameters()))[0])  # 打印加载后的参数
# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
SEQ_LEN = args['seq_len']

SCALER_X_PATH = r"D:\sea level variability\code_neao\convgru_TSVU - 反归一化\SOFTS-main\scaler_x_time.pkl"
SCALER_Y_PATH = r"D:\sea level variability\code_neao\convgru_TSVU - 反归一化\SOFTS-main\scaler_y_time.pkl"

X_NPY_PATH  = r"D:\sea level variability\DATA_neao\Anomalies_1993-2023.npy"
Y_XLSX_PATH = r"D:\sea level variability\DATA_neao\4processed_HIERRO_372.xlsx"


class CustomDataset(Dataset):
    def __init__(self):
        self.__load_data__()

    def __load_data__(self):
        self.scaler_x = joblib.load(SCALER_X_PATH)
        self.scaler_y = joblib.load(SCALER_Y_PATH)

        # X: (372,4,33,5,9) -> (372,5940)
        all_x = np.load(X_NPY_PATH)
        self.T = all_x.shape[0]
        all_x_2d = all_x.reshape(self.T, -1)  # ✅更稳，不写死 4*33*5*9
        self.data_x = self.scaler_x.transform(all_x_2d)

        # y：保留原始（含 NaN）——用于最终 Excel true
        df_y = pd.read_excel(Y_XLSX_PATH)
        y_raw = df_y.iloc[:, 1].values.reshape(-1, 1)
        self.y_data = y_raw  # ✅保留 NaN

        # y：模型输入需要填 NaN（仅用于输入，不用于输出 true）
        y_filled = y_raw.copy()
        mean_val = np.nanmean(y_filled)
        y_filled[np.isnan(y_filled)] = mean_val
        self.y_norm = self.scaler_y.transform(y_filled)

        # time
        time_data = df_y.iloc[:, 0].values
        dates = pd.DatetimeIndex(pd.to_datetime(time_data))
        self.data_stamp = time_features(dates, freq='D').transpose(1, 0)

    def __len__(self):
        return self.T - SEQ_LEN + 1  # window 数

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

# ✅ full_length 直接用原始序列长度（372）
FULL_LENGTH = test_data.T

# =========================
# 预测去重叠拼接（NEPO test 同款 sample_global_idx）
# =========================
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

        # ✅ 与你 NEPO 逻辑一致：MS -> f_dim=-1，否则 0
        f_dim = -1 if args['features'] == 'MS' else 0
        output = output[:, :, f_dim:]  # (B, L, 1) 通常

        out_np = output.detach().cpu().numpy()
        B, L, _ = out_np.shape  # L 理论上应为 12

        # ✅ 核心：窗口编号 = sample_global_idx，时间点 = start_idx + s
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

# 反归一化预测
pred_denorm = test_data.scaler_y.inverse_transform(pred_full)

# ✅ true：必须用原始 y（保留 NaN）
true_raw = test_data.y_data  # (372,1) 含 NaN

# 保存 Excel（true 不会被填充）
df = pd.DataFrame({
    "time_idx": np.arange(FULL_LENGTH),
    "true": true_raw.flatten(),              # ✅保留 NaN
    "pred": pred_denorm.flatten(),
    # 可选：把 counts 一并输出方便你检查覆盖次数（建议保留）
    "counts": counts
})
df.to_excel("NEAO_reconstructed_372_keep_nan_true.xlsx", index=False)

# 指标：只在非 NaN 位置算（避免 NaN 干扰）
mask = ~np.isnan(true_raw.flatten()) & ~np.isnan(pred_denorm.flatten())
rmse = np.sqrt(np.mean((true_raw.flatten()[mask] - pred_denorm.flatten()[mask]) ** 2))
mae  = np.mean(np.abs(true_raw.flatten()[mask] - pred_denorm.flatten()[mask]))
print(f"RMSE (non-NaN): {rmse:.4f}")
print(f"MAE  (non-NaN): {mae:.4f}")


# class CustomDataset(Dataset):
#     def __init__(self):
#         self.__loda_data__()
#
#     def __loda_data__(self):
#         self.scaler_x = joblib.load(r"D:\sea level variability\code_neao\convgru_TSVU - 反归一化\SOFTS-main\new_scaler_anomalies_2d.pkl")
#         self.scaler_y = joblib.load(r"D:\sea level variability\code_neao\convgru_TSVU - 反归一化\SOFTS-main\scaler_y_time.pkl")
#         # 加载372个时间点的X数据
#         file_path = r"D:\sea level variability\DATA_neao\Anomalies_1993-2023.npy"
#         all_x_data = np.load(file_path)
#         all_x_data_2d = all_x_data.reshape(-1, 4*33*5*9)
#         scaler = StandardScaler()
#         all_x_data_scaled = scaler.fit_transform(all_x_data_2d)
#         # 保存新的 scaler
#         joblib.dump(scaler, 'new_scaler_anomalies_2d.pkl')
#         self.data_x = all_x_data_scaled
#
#
#         # 加载y数据并归一化 372个时间点
#         y_path = r"D:\sea level variability\DATA_neao\4processed_HIERRO_372.xlsx"
#         self.y_data = pd.read_excel(y_path).iloc[:, 1].values.reshape(-1, 1)
#         y_series = pd.read_excel(y_path).iloc[:, 1]
#
#         # 处理空值：先前向填充，再用均值补齐剩余 NaN
#         y_filled = y_series.fillna(method='ffill')
#
#         # 若开头就有 NaN，ffill 无效 → 用均值填补
#         if y_filled.isna().sum() > 0:
#             mean_val = y_filled.mean()
#             y_filled = y_filled.fillna(mean_val)
#
#         # 转换为 numpy 格式并 reshape
#         y_data2 = y_filled.values.reshape(-1, 1)
#         # 2. 归一化
#         scaler_y = StandardScaler()
#         self.y_data_scaled = scaler_y.fit_transform(y_data2)
#         # 保存 scaler
#         joblib.dump(scaler_y, "scaler_y.pkl")
#
#         # 读取时间数据
#         time_path = r"D:\sea level variability\DATA_neao\4processed_HIERRO_372.xlsx"
#         time_data = pd.read_excel(time_path).iloc[:, 0].values
#         df_stamp = pd.to_datetime(time_data)
#         dates = pd.DatetimeIndex(df_stamp)
#         self.data_stamp = time_features(dates, freq='D')
#         self.data_stamp = self.data_stamp.transpose(1, 0)
#
#     def __len__(self):
#         return len(self.data_x) - args['seq_len'] + 1
#
#     def __getitem__(self, index):
#         if index >= len(self.data_stamp):
#             raise IndexError(f"Index {index} out of bounds for axis 0 with size {len(self.data_stamp)}")
#
#         seq_x = torch.tensor(self.data_x[index:index + args['seq_len']], dtype=torch.float32)  # 转换为 Tensor
#         seq_y = torch.tensor(self.y_data_scaled[index:index + args['seq_len']], dtype=torch.float32)  # 转换为 Tensor
#         seq_x_mark = torch.tensor(self.data_stamp[index:index + args['seq_len']], dtype=torch.float32)  # 转换为 Tensor
#         seq_y_mark = torch.tensor(self.data_stamp[index:index + args['seq_len']], dtype=torch.float32)  # 转换为 Tensor
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#
# #一个函数，创建数据集实例并返回数据集对象和对应的 DataLoader
# def data_provider():
#
#     def collate_fn(batch):
#         try:
#             seq_x_batch, seq_y_batch, batch_x_mark, batch_y_mark = zip(*batch)
#
#             seq_x_batch = torch.stack(seq_x_batch, dim=0)
#             seq_y_batch = torch.stack(seq_y_batch, dim=0)
#
#             batch_x_mark = [
#                 torch.tensor(np.array(item), dtype=torch.float32) if not isinstance(item, torch.Tensor) else item for
#                 item in batch_x_mark]
#             # batch_x_mark = [torch.tensor(item) if isinstance(item, list) else item for item in batch_x_mark]
#             batch_x_mark = torch.stack(batch_x_mark, dim=0)
#             # batch_y_mark = [torch.tensor(item) if isinstance(item, list) else item for item in batch_y_mark]
#             batch_y_mark = [
#                 torch.tensor(np.array(item), dtype=torch.float32) if not isinstance(item, torch.Tensor) else item for
#                 item in batch_y_mark]
#             batch_y_mark = torch.stack(batch_y_mark, dim=0)
#             return seq_x_batch, seq_y_batch, batch_x_mark, batch_y_mark
#         except Exception as e:
#             print("Batch processing error:", e)
#             raise
# #data_set 是一个 CustomDataset 类的实例，包含预加载的数据（self.data_x、self.y_data_normalized、self.data_stamp）
# # #data_set 是一个 CustomDataset 实例，封装了你的数据和访问逻辑。它是一个可迭代对象，DataLoader 会通过 __len__ 获取样本数，通过 __getitem__ 获取每个样本。
#     data_set=CustomDataset()
#     data_loader = DataLoader(
#         dataset=data_set,
#         batch_size=args['batch_size'],
#         shuffle=False,
#         num_workers=0,
#         drop_last=False, collate_fn=collate_fn)
#     #len(data_loader)=len(data_set)//batch_size+1,即451//32+1=15,即每个epoch有15个batch
#     return data_set, data_loader
#
# #调用 data_provider()，获取数据集和数据加载器，用于后续推理。test_data可以访问原始数据（test_data.data_x、test_data.y_data_normalized），test_loader实例用于迭代批次数据
# test_data, test_loader= data_provider()
#
# #切换到评估模式
# model.eval()
#
# # 推理并保存结果
# full_sequence = np.zeros((372, 1))  # 存储预测值
# full_y_true = np.zeros((372, 1))   # 存储真实值
# counts = np.zeros((372, 1))        # 记录覆盖次数
#
# all_outputs = []  # 用于存储所有反归一化预测
# full_sequence = np.zeros((372, 1))  # 目标序列：223 个时间步
# counts = np.zeros((372, 1))  # 记录每个时间步的覆盖次数
# with torch.no_grad():
#     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
#         batch_x = batch_x.float().to(device)
#         batch_y = batch_y.float().to(device)
#         batch_x_mark = batch_x_mark.float().to(device)
#         batch_y_mark = batch_y_mark.float().to(device)
#         #前向推理
#         output = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
#         print(f"Batch {i+1} model_outputs shape: {output.shape}")
#         # 转换为 numpy
#         outputs_cpu = output.cpu().numpy()  # (16, 12, 1)
#         batch_y_cpu = batch_y.cpu().numpy()  # (16, 12, 1)
#         # 反归一化
#         outputs_flat = outputs_cpu.reshape(-1, 1)
#         batch_y_flat = batch_y_cpu.reshape(-1, 1)
#         outputs_fgyh = test_data.scaler_y.inverse_transform(outputs_flat)  # 用 scaler_y_time.pkl 反归一化 output
#         batch_y_fgyh = test_data.scaler_y.inverse_transform(batch_y_flat)  # 用 scaler_y_time.pkl 反归一化 batch_y
#         print("scaler_y mean:", test_data.scaler_y.mean_)
#         print("scaler_y scale:", test_data.scaler_y.scale_)
#         print(f"Batch {i + 1}:")
#         print("Standardized output sample:", outputs_flat[:5].flatten())
#         print("Inverse transformed output sample:", outputs_fgyh[:5].flatten())
#         # 恢复形状
#         outputs_original = outputs_fgyh.reshape(outputs_cpu.shape)  # (16, 12, 1)
#         batch_y_original = batch_y_fgyh.reshape(batch_y_cpu.shape)  # (16, 12, 1)
#         print("Model outputs range (before inverse_transform):", np.min(outputs_cpu), np.max(outputs_cpu))
#         # 映射到完整时间轴
#         batch_size = outputs_original.shape[0]
#         start_idx = i * args['batch_size']
#         pred_len = args['pred_len']
#
#         for j in range(batch_size):
#             pred_start = start_idx + j
#             pred_end = pred_start + pred_len
#             valid_end = min(pred_end,372)
#             slice_len = valid_end - pred_start
#
#             # 累加预测值和真实值
#             full_sequence[pred_start:valid_end] += outputs_original[j, :slice_len, 0].reshape(-1, 1)
#             full_y_true[pred_start:valid_end] += batch_y_original[j, :slice_len, 0].reshape(-1, 1)
#             counts[pred_start:valid_end] += 1
#     full_sequence = full_sequence / np.maximum(counts, 1)
#     # print("Predicted values range (full_sequence):", np.min(full_sequence), np.max(full_sequence))
#     # 恢复原始 y_data 的 NaN
#     full_y_true_raw = test_data.y_data  # 直接使用原始 y_data，包含 NaN
#     full_y_true = full_y_true / np.maximum(counts, 1)  # 计算平均后的 y_true
#
#     # 保存到 DataFrame
#     df = pd.DataFrame({
#         'True_Values': full_y_true_raw.flatten(),  # 保留 NaN 的原始真实值
#         # 'True_Values_Averaged': full_y_true.flatten(),  # 平均后的真实值
#         'Predicted_Values': full_sequence.flatten()
#     })
#     df.to_excel('reconstructed_372_timesteps_with_true.xlsx', index=False)
#
#     # 计算 RMSE（仅非 NaN 点）
#     mask = ~np.isnan(full_y_true_raw) & ~np.isnan(full_sequence)
#     rmse = np.sqrt(np.mean((full_y_true_raw[mask] - full_sequence[mask]) ** 2))
#     # 计算 MAE（非 NaN 点）
#     mae = np.mean(np.abs(full_y_true_raw[mask] - full_sequence[mask]))
#
#     # 同时打印 RMSE 和 MAE
#     print(f"RMSE (non-NaN points, window average): {rmse:.4f}")
#     print(f"MAE  (non-NaN points, window average): {mae:.4f}")
    # print("Reconstructed shape:", full_sequence.shape)
    # print("True values shape:", full_y_true_raw.shape)