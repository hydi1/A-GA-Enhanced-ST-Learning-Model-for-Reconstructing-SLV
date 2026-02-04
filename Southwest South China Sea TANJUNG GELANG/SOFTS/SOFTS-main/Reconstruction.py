import pandas as pd
import numpy as np
import pickle
import joblib
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
from torch.utils.data import Dataset
from models.SOFTS import Model  # 根据文件和类的实际名称调整路径
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
        # 'loss': 'MSE',
        "use_norm": False,
        'd_core': 512,
        'freq': 'D',
        'root_path': r'D:\sea level variability\DATA_eio\1589',
        'data_path': "anomaly_1993_2021_depth15_filtered.npy",
        'target_path': r"D:\sea level variability\DATA_eio\1589\processed_1589.xlsx",
        'target': "OT",  # OT 是什么意思
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention':False,
        "lradj": "type1",
        # 'learning_rate': 0.0001,
        'checkpoints': r'D:\sea level variability\code_eio\SOFTS_TS -12\SOFTS-main\checkpoints',
        "save_model":True,
        'device_ids':[0],
        'scale': True,

        }


# ================= 2. 数据集类定义 =================
class CustomDataset(Dataset):
    def __init__(self):
        self.__loda_data__()


    def __loda_data__(self):
        # 1. 加载372个时间点的X数据 (1993-2023)
        file_path = r"D:\sea level variability\DATA_eio\1589\anomaly_1993_2023_depth15_372time.npy"
        all_x_data = np.load(file_path)
        # 展平为 (372, 9900)
        all_x_data_2d = all_x_data.reshape(-1, 4 * 15 * 11 * 15)

        # 2. 检查数据类型与无效值处理
        if all_x_data_2d.dtype != np.float64:
            print("警告：输入数据类型为 {}, 转换为 np.float64".format(all_x_data_2d.dtype))
            all_x_data_2d = all_x_data_2d.astype(np.float64)

        # 计算全局均值用于填充（避开所有无效值）
        global_mean = np.nanmean(all_x_data_2d[np.isfinite(all_x_data_2d)])

        if np.any(np.isnan(all_x_data_2d)):
            print("警告：输入数据包含 NaN，已用均值填充")
            all_x_data_2d = np.nan_to_num(all_x_data_2d, nan=global_mean)

        if np.any(np.isinf(all_x_data_2d)):
            print("警告：输入数据包含 Inf，已用均值填充")
            all_x_data_2d = np.nan_to_num(all_x_data_2d, posinf=global_mean, neginf=global_mean)

        # 3. 检查数据是否包含离谱极值 (如 9.99e37)
        if np.any(np.abs(all_x_data_2d) > 1e8):
            print("警告：输入数据包含极端值，正在执行鲁棒标准化")
            # 针对极值进行初步的 Z-Score 处理
            mean_tmp = np.nanmean(all_x_data_2d, axis=0)
            std_tmp = np.nanstd(all_x_data_2d, axis=0)
            all_x_data_2d = (all_x_data_2d - mean_tmp) / (std_tmp + 1e-8)

        # 4. 归一化输入数据 (生成新的 Scaler)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        all_x_data_scaled = scaler.fit_transform(all_x_data_2d)

        # 5. 归一化后的二次安全检查
        if np.any(np.isnan(all_x_data_scaled)) or np.any(np.isinf(all_x_data_scaled)):
            print("错误：归一化后的数据依然包含 NaN 或 Inf，尝试强制清零")
            all_x_data_scaled = np.nan_to_num(all_x_data_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # 保存并重新加载 scaler (确保与 self.scaler_x 属性一致)
        scaler_save_path = r'D:\sea level variability\code_eio\convgru_TSVU - 反归一化12\SOFTS-main\new_scaler_anomalies_2d.pkl'
        joblib.dump(scaler, scaler_save_path)
        self.scaler_x = joblib.load(scaler_save_path)
        self.data_x = all_x_data_scaled

        # 6. 加载 y 数据并归一化
        y_path = r"D:\sea level variability\DATA_eio\1589\processed_1589 - 372.xlsx"
        df_y = pd.read_excel(y_path)
        y_data = df_y.iloc[:, 1].values.reshape(-1, 1)
        self.y_data = y_data  # 保留原始数据 (含 NaN)

        # y 缺失值填充并使用旧的 scaler_y 转换
        y_mean = np.nanmean(y_data)
        y_data_filled = np.where(np.isnan(y_data), y_mean, y_data)
        self.scaler_y = joblib.load(
            r"D:\sea level variability\code_eio\SOFTS_TS -12\SOFTS-main\scaler_y_time.pkl")
        self.y_data_normalized = self.scaler_y.transform(y_data_filled)

        # 7. 读取时间数据
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


# ================= 3. 数据提供者函数 =================
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


# ================= 4. 模型初始化与加载 =================
# 自动根据 args['model'] 加载对应的 GRU 类
model = Model(args)

checkpoint_path = r"D:\sea level variability\code_eio\SOFTS_TS -12\SOFTS-main\checkpoints\TSdepth51993-2023_345_SOFTS_ssta_MS_0.0005_12_12_12_64_2_1_256\checkpoint.pth"
state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# ================= 5. 推理与滑动窗口平均 (修正版) =================
test_data, test_loader = data_provider()

# 显式定义总长度，确保与 y_data 一致
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

        # 前向推理
        output = model(batch_x, batch_x_mark, batch_y, batch_y_mark)

        # 将输出转回 CPU 并反归一化
        outputs_cpu = output.detach().cpu().numpy()  # 形状: (Batch, Pred_len, 1)

        # 执行反归一化
        # 注意：这里需要先 flatten 再 transform，最后还原形状
        outputs_flat = outputs_cpu.reshape(-1, 1)
        outputs_unnorm = test_data.scaler_y.inverse_transform(outputs_flat)
        outputs_original = outputs_unnorm.reshape(outputs_cpu.shape)

        # 获取当前 Batch 的实际大小（最后一个 batch 可能小于 16）
        current_batch_size = outputs_original.shape[0]
        start_idx = i * args['batch_size']

        # 将每个窗口的预测结果累加到总序列中
        for j in range(current_batch_size):
            p_start = start_idx + j
            p_end = p_start + args['pred_len']

            # 确保预测范围不超过 372
            v_end = min(p_end, total_steps)
            s_len = v_end - p_start

            # 仅累加有效长度部分的数据
            if s_len > 0:
                full_sequence[p_start:v_end] += outputs_original[j, :s_len, 0].reshape(-1, 1)
                counts[p_start:v_end] += 1

# 计算窗口覆盖后的平均值，防止最后一部分因为没有覆盖而变成 NaN
# 使用 np.where 将 counts 为 0 的地方设为 1，避免除以 0，同时保留初始化的 0
final_output = full_sequence / np.where(counts == 0, 1, counts)

# 调试打印：检查最后 10 个点的覆盖情况
print(f"最后 10 个点的覆盖次数: {counts[-10:].flatten()}")
print(f"最后 10 个点的预测值是否包含 NaN: {np.isnan(final_output[-10:]).any()}")

# ================= 6. 保存结果 =================
full_y_true_raw = test_data.y_data  # 包含原始 NaN 的数据 (372, 1)

df_res = pd.DataFrame({
    'True_Values': full_y_true_raw.flatten(),
    'Predicted_Values': final_output.flatten()
})

df_res.to_excel('reconstructed_372_timesteps_with_true2SOFTS.xlsx', index=False)
print("保存成功！文件包含 372 行数据。")