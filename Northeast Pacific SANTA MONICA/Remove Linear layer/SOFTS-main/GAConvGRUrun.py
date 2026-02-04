import numpy as np
import pandas as pd
import torch
import random
from sklearn.model_selection import train_test_split
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from torchsummary import summary

def count_param(model):
    """计算模型的总可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate_model(seed=42):
    """
    Train and evaluate the SOFTS model using the provided training, validation, and test data.
    """
    # Configure arguments for the experiment
    args = {
        'task_name': 'uvst',
        'model_id':  f"seed{seed}",
        'model': 'GAConvGRU',
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
        'checkpoints': r'D:\project\组件消融\convgru-移除Linear\SOFTS-main\checkpoints',
        "save_model": True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }

    # 设置随机种子以确保可重复性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 初始化实验
    exp = Exp_Long_Term_Forecast(args)
    # 开始训练
    print(f"开始训练，模型 ID: {args['model_id']}, 种子: {seed}")
    exp.train(args)
    print("训练完成！")

    # 在验证集上评估
    print("开始在验证集上评估...")
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
    result = exp.test(setting)

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

    # 开始训练
    exp.train(args)
    print("训练完成！")

    # 在测试集上评估
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

    # 获取 test 返回的指标
    result = exp.test(setting)
    rmse_norm_full = result.get('rmse_norm_full')
    mae_norm_full = result.get('mae_norm_full')
    r2_eff_norm_full = result.get('r2_eff_norm_full')
    rmse_denorm_full = result.get('rmse_denorm_full')
    mae_denorm_full = result.get('mae_denorm_full')

    print(f"\n种子 {seed} 评估完成！")
    print(f"[Normalized]    RMSE: {rmse_norm_full:.4f}, MAE: {mae_norm_full:.4f}, R2_eff: {r2_eff_norm_full:.4f}")
    print(f"[De-normalized] RMSE: {rmse_denorm_full:.4f}, MAE: {mae_denorm_full:.4f}")

    return rmse_norm_full, mae_norm_full, r2_eff_norm_full, rmse_denorm_full, mae_denorm_full

if __name__ == "__main__":
    # 测试多个种子100, 200, 300, 400, 500,
    #         123, 234, 345, 456, 567,
    #         678, 789, 888, 999, 1000,
    #         1111, 1222, 1333, 1444, 1555,
    #         1666, 1777, 1888, 1999, 2024,
    #         2025, 2048, 2077, 2099, 2121,
    #         2222, 2333, 2444, 2555, 2666,
    #         2777, 2888, 2999, 3001, 3333,
    #         3456, 3579, 3690, 4040, 5050
    seed_list = [42, 43, 44, 45, 46,
                 100, 200, 300, 400, 500,
                123, 234, 345, 456, 567,
                678, 789, 888, 999, 1000,
        ]  # 测试1个种子，可根据需要增加更多种子
    
    results = []
    for seed in seed_list:
        # 接收 test 返回的5个指标
        rmse_norm, mae_norm, r2_eff_norm_full, rmse_denorm, mae_denorm = train_and_evaluate_model(seed=seed)
        results.append((seed, rmse_norm, mae_norm, r2_eff_norm_full, rmse_denorm, mae_denorm))

    # --- 性能分析与输出 ---
    # 1. 按 RMSE(norm) 升序排序，找到最好的种子（RMSE 越小越好）
    results.sort(key=lambda x: x[1])
    # results entries: (seed, rmse_norm, mae_norm, r2_eff_norm_full, rmse_denorm, mae_denorm)
    best_seed, b_rmse_norm, b_mae_norm, b_r2_eff, b_rmse_denorm, b_mae_denorm = results[0]

    print("\n" + "=" * 110)
    print(f"{'所有种子测试结果 (按 RMSE(norm) 排序)':^110}")
    print("-" * 110)
    print(f"{'种子':<8} | {'RMSE(norm)':<12} | {'MAE(norm)':<12} | {'R2_eff(norm)':<16} | {'RMSE(denorm)':<14} | {'MAE(denorm)':<12}")
    print("-" * 110)
    for seed, rn, mn, r2eff, rd, md in results:
        print(f"{seed:<8} | {rn:<12.4f} | {mn:<12.4f} | {r2eff:<16.4f} | {rd:<14.4f} | {md:<12.4f}")
    print("-" * 110)

    print(f"\n🥇 最佳种子: {best_seed}")
    print(f"  > [Normalized]    RMSE: {b_rmse_norm:.4f}, MAE: {b_mae_norm:.4f}, R2_eff: {b_r2_eff:.4f}")
    print(f"  > [De-normalized] RMSE: {b_rmse_denorm:.4f}, MAE: {b_mae_denorm:.4f}")

    # 2. 计算所有性能指标的平均性能
    rmse_norms = np.array([r[1] for r in results])
    mae_norms = np.array([r[2] for r in results])
    r2_effs = np.array([r[3] for r in results])
    rmse_denorms = np.array([r[4] for r in results])
    mae_denorms = np.array([r[5] for r in results])

    print("\n" + "=" * 70)
    print(f"{'平均性能统计 (所有种子)':^70}")
    print("-" * 70)
    print(f"[Normalized+去重叠] RMSE: {np.mean(rmse_norms):.4f} ± {np.std(rmse_norms, ddof=1):.4f}")
    print(f"[Normalized+去重叠] MAE:  {np.mean(mae_norms):.4f} ± {np.std(mae_norms, ddof=1):.4f}")
    print(f"[Normalized+去重叠] R2_eff: {np.mean(r2_effs):.4f} ± {np.std(r2_effs, ddof=1):.4f}")
    print(f"\n[De-normalized] RMSE: {np.mean(rmse_denorms):.4f} ± {np.std(rmse_denorms, ddof=1):.4f}")
    print(f"[De-normalized] MAE:  {np.mean(mae_denorms):.4f} ± {np.std(mae_denorms, ddof=1):.4f}")
    print("=" * 70)