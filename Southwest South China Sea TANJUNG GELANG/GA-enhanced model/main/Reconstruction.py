import pandas as pd
import numpy as np
import pickle
import joblib
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
from torch.utils.data import Dataset
from models.GAconvgru import Model  # 根据文件和类的实际名称调整路径
from sklearn.preprocessing import StandardScaler
import joblib
from utils.timefeatures import time_features

args = {
    'task_name': 'GAConvgru_610',
    'model_id': 9499,
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
    'input_size': 1320,
    'hidden_size': 1320,
    'output_size': 1,
    'num_layers': 3,
    'root_path': r'D:\sea level variability\DATA_eio\1589',
    'data_path': "anomaly_1993_2018_depth15_filtered.npy",  # 数据形状 (shape): (306, 4, 15, 11, 15)
    'target_path': r"D:\sea level variability\DATA_eio\1589\processed_1589.xlsx",
    'target': "OT",  # OT 可能是目标变量名称（如 Ocean Temperature），需确认
    'seasonal_patterns': 'Monthly',
    'num_workers': 4,
    'use_amp': False,
    'output_attention': False,
    "lradj": "type1",
    'checkpoints': r'D:\sea level variability\code_eio\convgru_TSVU - 反归一化\SOFTS-main\checkpoints',
    'save_model': True,
    'device_ids': [0],
    'scale': True,
    'num_heads': 4,
}

# 假设模型初始化
model = Model(args)

# 加载模型参数
state_dict = torch.load(r"D:\xwechat_files\wxid_zvtu4tfj0vaj22_4045\msg\file\2026-01\GAConvgru_seed9175_GAconvgru_ssta_MS_0.0005_12_12_12_64_2_1_256\GAConvgru_seed9175_GAconvgru_ssta_MS_0.0005_12_12_12_64_2_1_256\checkpoint.pth",weights_only=True)
# state_dict = torch.load(r"D:\sea level variability\code_eio\convgru_TSVU - 反归一化\SOFTS-main\checkpoints\GAConvgru_610_seed2024_GAconvgru_ssta_MS_0.0001_12_12_12_64_2_1_256\checkpoint.pth",weights_only=True)
# 移除 'module.' 前缀
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
# print("Before loading, first parameter sample:", next(iter(model.parameters()))[0])  # 打印初始参数
# 加载更新后的 state_dict
model.load_state_dict(state_dict)
# print("After loading, first parameter sample:", next(iter(model.parameters()))[0])  # 打印加载后的参数
# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
class CustomDataset(Dataset):
    def __init__(self):
        self.__loda_data__()
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

class CustomDataset:
    def __init__(self):
        self.__loda_data__()

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

class CustomDataset:
    def __init__(self):
        self.__loda_data__()

    def __loda_data__(self):
        # 加载372个时间点的X数据1993-2023
        file_path = r"D:\sea level variability\DATA_eio\1589\anomaly_1993_2023_depth15_372time.npy"
        all_x_data = np.load(file_path)
        all_x_data_2d = all_x_data.reshape(-1, 4*15*11*15)

        # 检查数据类型
        if all_x_data_2d.dtype != np.float64:
            print("警告：输入数据类型为 {}, 转换为 np.float64".format(all_x_data_2d.dtype))
            all_x_data_2d = all_x_data_2d.astype(np.float64)

        # 检查输入数据是否包含无效值
        if np.any(np.isnan(all_x_data_2d)):
            print("警告：输入数据包含 NaN，已用均值填充")
            all_x_data_2d = np.nan_to_num(all_x_data_2d, nan=np.nanmean(all_x_data_2d))
        if np.any(np.isinf(all_x_data_2d)):
            print("警告：输入数据包含 Inf，已用均值填充")
            all_x_data_2d = np.nan_to_num(all_x_data_2d, posinf=np.nanmean(all_x_data_2d), neginf=np.nanmean(all_x_data_2d))

        # 检查数据是否包含极值
        if np.any(np.abs(all_x_data_2d) > 1e8):
            print("警告：输入数据包含极端值，已标准化")
            mean = np.nanmean(all_x_data_2d, axis=0)
            std = np.nanstd(all_x_data_2d, axis=0)
            all_x_data_2d = (all_x_data_2d - mean) / (std + 1e-8)  # 避免除以零

        # 归一化输入数据
        scaler = StandardScaler()
        all_x_data_scaled = scaler.fit_transform(all_x_data_2d)

        # 检查归一化后的数据是否包含无效值
        if np.any(np.isnan(all_x_data_scaled)):
            print("错误：归一化后的数据包含 NaN")
            raise ValueError("归一化后的数据包含 NaN")
        if np.any(np.isinf(all_x_data_scaled)):
            print("错误：归一化后的数据包含 Inf")
            raise ValueError("归一化后的数据包含 Inf")

        joblib.dump(scaler, 'new_scaler_anomalies_2d.pkl')
        self.scaler_x = joblib.load(r'D:\sea level variability\code_eio\convgru_TSVU - 反归一化12\SOFTS-main\new_scaler_anomalies_2d.pkl')

        self.data_x = all_x_data_scaled

        # 加载y数据并归一化
        y_path = r"D:\sea level variability\DATA_eio\1589\processed_1589 - 372.xlsx"
        y_data = pd.read_excel(y_path).iloc[:, 1].values.reshape(-1, 1)
        self.y_data = y_data  # 保留原始数据，包含 NaN
        y_data_filled = np.where(np.isnan(y_data), np.nanmean(y_data), y_data)
        self.scaler_y = joblib.load(r"D:\sea level variability\code_eio\convgru_TSVU - 反归一化12\SOFTS-main\scaler_y_time.pkl")
        self.y_data_normalized = self.scaler_y.transform(y_data_filled)

        # 读取时间数据
        time_path = r"D:\sea level variability\DATA_eio\1589\processed_1589 - 372.xlsx"
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

    data_set = CustomDataset()
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=0,
        drop_last=False, collate_fn=collate_fn)
    return data_set, data_loader

test_data, test_loader = data_provider()
model.eval()
full_sequence = np.zeros((372, 1))
full_y_true = np.zeros((372, 1))
counts = np.zeros((372, 1))

with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # 检查输入
        if torch.any(torch.isnan(batch_x)) or torch.any(torch.isinf(batch_x)):
            print(f"批次 {i + 1} batch_x 包含 NaN 或 Inf")

        # 前向推理
        output = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        print(f"Batch {i + 1} model_outputs shape: {output.shape}")
        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            print(f"批次 {i + 1} 模型输出包含 NaN 或 Inf")

        # 转换为 numpy
        outputs_cpu = output.cpu().numpy()
        batch_y_cpu = batch_y.cpu().numpy()
        outputs_flat = outputs_cpu.reshape(-1, 1)
        batch_y_flat = batch_y_cpu.reshape(-1, 1)

        # 反归一化
        outputs_fgyh = test_data.scaler_y.inverse_transform(outputs_flat)
        batch_y_fgyh = test_data.scaler_y.inverse_transform(batch_y_flat)
        if np.any(np.isnan(outputs_fgyh)):
            print(f"批次 {i + 1} 反归一化后存在 NaN 值")
            print(f"问题值 (outputs_flat): {outputs_flat[np.isnan(outputs_fgyh)]}")
            outputs_fgyh = np.nan_to_num(outputs_fgyh, nan=0.0)  # 替换 NaN

        outputs_original = outputs_fgyh.reshape(outputs_cpu.shape)
        batch_y_original = batch_y_fgyh.reshape(batch_y_cpu.shape)

        # 映射到时间轴
        batch_size = outputs_original.shape[0]
        start_idx = i * args['batch_size']
        for j in range(batch_size):
            pred_start = start_idx + j
            pred_end = pred_start + args['pred_len']
            valid_end = min(pred_end, 372)
            slice_len = valid_end - pred_start
            if slice_len > 0:
                full_sequence[pred_start:valid_end] += outputs_original[j, :slice_len, 0].reshape(-1, 1)
                full_y_true[pred_start:valid_end] += batch_y_original[j, :slice_len, 0].reshape(-1, 1)
                counts[pred_start:valid_end] += 1
                print(f"批次 {i + 1}, 序列 {j}, 填充时间步: {pred_start} 到 {valid_end - 1}")

    # 平均预测值
    full_sequence = full_sequence / np.maximum(counts, 1)
    full_y_true = full_y_true / np.maximum(counts, 1)

    # 保存结果到 Excel
    df = pd.DataFrame({
        'True_Values': test_data.y_data.flatten(),
        'Predicted_Values': full_sequence.flatten()
    })
    df.to_excel('reconstructed_372_timesteps_with_true_seed2024.xlsx', index=False)

    # ====================== 正确的 RMSE + MAE 计算 ======================
    # 先统一转成一维数组（强烈推荐这样写，永不出错）
    true_flat = test_data.y_data.flatten()  # shape: (372,)
    pred_flat = full_sequence.flatten()  # shape: (372,)

    # 生成一维的掩码，只保留真实值非 NaN 的位置（预测值一般不会有 NaN）
    mask = ~np.isnan(true_flat)

    # 取出有效数据
    true_valid = true_flat[mask]
    pred_valid = pred_flat[mask]

    # 计算 RMSE 和 MAE
    rmse = np.sqrt(np.mean((true_valid - pred_valid) ** 2))
    mae = np.mean(np.abs(true_valid - pred_valid))

    # 打印结果
    print(f"RMSE (non-NaN points): {rmse:.4f}")
    print(f"MAE  (non-NaN points): {mae:.4f}")
    # =====================================================================
#切换到评估模式
# model.eval()
#
# # 推理并保存结果
# full_sequence = np.zeros((372, 1))  # 存储预测值
# full_y_true = np.zeros((372, 1))   # 存储真实值
# counts = np.zeros((372, 1))        # 记录覆盖次数
#
# all_outputs = []  # 用于存储所有反归一化预测
# full_sequence = np.zeros((372, 1))  # 目标序列：372 个时间步
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
#         if np.any(np.isnan(outputs_fgyh)):
#             print(f"批次 {i + 1} 反归一化后存在 NaN 值")
#         batch_y_fgyh = test_data.scaler_y.inverse_transform(batch_y_flat)  # 用 scaler_y_time.pkl 反归一化 batch_y
#         # 恢复形状
#         outputs_original = outputs_fgyh.reshape(outputs_cpu.shape)  # (16, 12, 1)
#         batch_y_original = batch_y_fgyh.reshape(batch_y_cpu.shape)  # (16, 12, 1)
#
#         # 映射到完整时间轴
#         batch_size = outputs_original.shape[0]
#         start_idx = i * args['batch_size']
#         pred_len = args['pred_len']
#
#         for j in range(batch_size):
#             pred_start = start_idx + j
#             pred_end = pred_start + pred_len
#             valid_end = min(pred_end, 372)
#             slice_len = valid_end - pred_start
#
#             # 累加预测值和真实值
#             full_sequence[pred_start:valid_end] += outputs_original[j, :slice_len, 0].reshape(-1, 1)
#             full_y_true[pred_start:valid_end] += batch_y_original[j, :slice_len, 0].reshape(-1, 1)
#             counts[pred_start:valid_end] += 1
#
#     # # 计算平均值
#     # full_sequence = full_sequence / np.maximum(counts, 1)  # 预测值平均
#     # full_y_true = full_y_true / np.maximum(counts, 1)  # 真实值平均
#     #
#     # # 保存到 DataFrame
#     # df = pd.DataFrame({
#     #     'True_Values': full_y_true.flatten(),  # 反归一化的真实值
#     #     'Predicted_Values': full_sequence.flatten()  # 反归一化的预测值
#     # })
#     # df.to_excel('reconstructed_372_timesteps_with_true.xlsx', index=False)
#     # print("Reconstructed shape:", full_sequence.shape)
#     # print("True values shape:", full_y_true.shape)
#     # 计算平均值，仅对非 NaN 点
#     full_sequence = full_sequence / np.maximum(counts, 1)
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
#     df.to_excel('reconstructed_372_timesteps_with_true2.xlsx', index=False)
#
#     # 计算 RMSE（仅非 NaN 点）
#     mask = ~np.isnan(full_y_true_raw) & ~np.isnan(full_sequence)
#     rmse = np.sqrt(np.mean((full_y_true_raw[mask] - full_sequence[mask]) ** 2))
#     print(f"RMSE (non-NaN points): {rmse:.4f}")
#     print("Reconstructed shape:", full_sequence.shape)
#     print("True values shape:", full_y_true_raw.shape)