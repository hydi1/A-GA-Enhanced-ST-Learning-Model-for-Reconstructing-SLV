import numpy as np
import pandas as pd
import torch
import random
from sklearn.model_selection import train_test_split
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from torchsummary import summary


def count_param(model):
    """计算模型的可训练参数总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate_model(seed=42):
    """
    Train and evaluate the SOFTS model using the provided training, validation, and test data.
    """
    # set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Configure arguments for the experiment
    args = {
        'task_name': 'vst',
        'model_id': f"seed{seed}",
        'model': 'GAconvGRU',
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
        'input_size': 1056,
        'hidden_size': 64,
        'output_size': 1,  # 假设是回归任务，输出一个预测值
        'num_layers': 3,  # 使用3层GRU
        'root_path': r'D:\goole\2025115',
        "data_path": 'TSuv_data_368.npy',
        "target_path": r"D:\goole\GOPRdata\Y非nan -1993_2023.xlsx",
        'target': "OT",  # OT 可能是目标变量名称（如 Ocean Temperature），需确认
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        "lradj": "type1",
        # 'learning_rate': 0.0001,
        'checkpoints': r'D:\project\组件消融\convgru-移除path1\SOFTS-main\checkpoints',
        "save_model": True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }

    # 初始化实验
    exp = Exp_Long_Term_Forecast(args)
    print(f"开始训练，模型 ID: {args['model_id']}, 种子: {seed}")

    # 显式构建模型以访问它
    model = exp._build_model()

    # 计算并打印参数量
    print("总可训练参数量：", count_param(model))

    # 获取一个真实的输入批次
    train_data, train_loader = exp._get_data(flag='train')
    batch = next(iter(train_loader))
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch

    # 移动到与模型相同的设备
    device = torch.device('cuda' if args['use_gpu'] and torch.cuda.is_available() else 'cpu')
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device) if batch_x_mark is not None else None
    batch_y = batch_y.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device) if batch_y_mark is not None else None

    # 构造 x_dec（根据你的 forward 方法逻辑）
    dec_inp = batch_y

    # 使用 input_data 传递实际输入
    input_data = (batch_x, batch_x_mark, dec_inp, batch_y_mark)
    # print(summary(model, input_data=input_data))

    exp.train(args)
    print("训练完成！")
    # 在验证集上评估
    print("开始在测试集上评估 (去重叠全局指标)...")
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args['task_name'],
        args['model_id'],
        args['model'],
        args['data'],
        args['features'],
        args['seq_len'],
        args['label_len'],
        args['pred_len'],
        args['d_model'],
        args['e_layers'],
        args['d_layers'],
        args['d_ff'],
        args['factor'],
        args['embed'],
        args['distil'],
        args['target']
    )

    # 获取 test 返回的指标（包含归一化与反归一化）
    result = exp.test(setting)
    rmse_norm = result.get('rmse_norm', None)
    mae_norm = result.get('mae_norm', None)
    r2_eff_norm_full = result.get('r2_eff_norm_full', None)
    rmse = result.get('rmse', None)
    mae = result.get('mae', None)

    print(f"种子 {seed} 评估完成！RMSE(norm): {rmse_norm:.4f}, MAE(norm): {mae_norm:.4f}, R2_eff(norm,full): {r2_eff_norm_full:.4f} | RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse_norm, mae_norm, r2_eff_norm_full, rmse, mae

if __name__ == "__main__":

#     1111, 1222, 1333, 1444, 1555,
#     1666, 1777, 1888, 1999, 2024,
#     2025, 2048, 2077, 2099, 2121,
# 2222, 2333, 2444, 2555, 2666,
# 2777, 2888, 2999, 3001, 3333,
# 3456, 3579, 3690, 4040, 5050
    seed_list = [
            42, 43, 44, 45, 46,
            100, 200, 300, 400, 500,
             123, 234, 345, 456, 567,
            678, 789, 888, 999, 1000,

    ]

    results = []
    for seed in seed_list:
        # 接收 test 返回的五个指标（rmse_norm, mae_norm, r2_eff_norm_full, rmse, mae）
        rmse_norm, mae_norm, r2_eff_full_norm, rmse, mae = train_and_evaluate_model(seed=seed)
        results.append((seed, rmse_norm, mae_norm, r2_eff_full_norm, rmse, mae))
    # --- 性能分析与输出 ---
    # 1. 按 RMSE(norm) 升序排序，找到最好的种子（RMSE 越小越好）
    results.sort(key=lambda x: x[1])
    # results entries: (seed, rmse_norm, mae_norm, r2_eff_norm_full, rmse, mae)
    best_seed, b_rmse_norm, b_mae_norm, b_r2_eff, b_rmse, b_mae = results[0]

    print("\n" + "="*95)
    print(f"{'所有种子测试结果 (按 RMSE(norm) 排序)':^95}")
    print("-" * 95)
    print(f"{'种子':<8} | {'RMSE(norm)':<12} | {'MAE(norm)':<12} | {'R2_eff(norm,full)':<16} | {'RMSE':<12} | {'MAE':<12}")
    print("-" * 95)
    for seed, rn, mn, r2eff, r, m in results:
        print(f"{seed:<8} | {rn:<12.4f} | {mn:<12.4f} | {r2eff:<16.4f} | {r:<12.4f} | {m:<12.4f}")
    print("-" * 95)

    print(f"🥇 最佳种子: {best_seed}")
    print(f"  > RMSE(norm): {b_rmse_norm:.4f}, MAE(norm): {b_mae_norm:.4f}, R2_eff(norm,full): {b_r2_eff:.4f}")
    print(f"  > RMSE: {b_rmse:.4f}, MAE: {b_mae:.4f}")

    # 2. 计算所有性能指标的平均性能
    # 分别计算归一化与反归一化指标的统计
    rmse_norms = [r[1] for r in results]
    mae_norms = [r[2] for r in results]
    r2_effs = [r[3] for r in results]
    rmses = [r[4] for r in results]
    maes = [r[5] for r in results]

    print("\n" + "="*50)
    print(f"{'平均性能统计 (所有种子)':^50}")
    print("-" * 50)
    print(f"归一化+去重叠RMSE: {np.mean(rmse_norms):.4f} ± {np.std(rmse_norms, ddof=1):.4f}")
    print(f"归一化+去重叠 MAE:  {np.mean(mae_norms):.4f} ± {np.std(mae_norms, ddof=1):.4f}")
    print(f"R2_eff(归一化+去重叠): {np.mean(r2_effs):.4f} ± {np.std(r2_effs, ddof=1):.4f}")
    print(f"反归一化 RMSE: {np.mean(rmses):.4f} ± {np.std(rmses, ddof=1):.4f}")
    print(f"反归一化 MAE:  {np.mean(maes):.4f} ± {np.std(maes, ddof=1):.4f}")
    print("="*50)