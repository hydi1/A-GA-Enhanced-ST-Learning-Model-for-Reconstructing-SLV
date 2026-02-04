
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from torchinfo import summary
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


# 定义计算参数量的函数
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def set_seed(seed):
    """
    设置随机种子以确保可重复性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_and_evaluate_model(seed=42):
    """
    使用提供的训练、验证和测试数据训练并评估 SOFTS 模型
    """
    # 1. 首先设置随机种子
    set_seed(seed)

    # 2. 配置实验参数
    args = {
        'task_name': 'uvst',
        'model_id': f"seed{seed}",
        'model': 'ConvGRU',
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
        'input_size': 1056,
        'hidden_size': 1056,
        'output_size': 1,
        'num_layers': 3,
        'root_path': r'D:\goole\2025115',
        'data_path': 'TSuv_data_368.npy',
        'target_path': r'D:\goole\GOPRdata\Y非nan -1993_2023.xlsx',
        'target': 'OT',
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        'record_grad': False,
        'lradj': 'type1',
        'checkpoints': r'D:\project\组件消融\东北太平洋\GAconvgru替换为convgru\SOFTS-main\checkpoints',
        'save_model': True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }

    # 3. 初始化实验
    exp = Exp_Long_Term_Forecast(args)
    print(f"\n>>>>>>> 开始训练，模型: {args['model']}, 种子: {seed} <<<<<<<")
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
    print(summary(model, input_data=input_data))
    # 4. 训练模型
    exp.train(args)
    print("训练完成！")

    # 5. 评估模型
    print("开始在测试集上评估 (去重叠后的反归一化全局指标)...")
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args['task_name'], args['model_id'], args['model'], args['data'], args['features'],
        args['seq_len'], args['label_len'], args['pred_len'], args['d_model'], args['e_layers'],
        args['d_layers'], args['d_ff'], args['factor'], args['embed'], args['distil'], args['target']
    )

    # 获取 test 返回的全局指标（包含归一化和未归一化）
    result = exp.test(setting)
    rmse_norm = result.get('rmse_norm')
    mae_norm = result.get('mae_norm')
    r2_eff_norm_full = result.get('r2_eff_norm_full')
    rmse = result.get('rmse')
    mae = result.get('mae')

    print(f"种子 {seed} 评估完成！[Normalized] RMSE: {rmse_norm:.4f}, MAE: {mae_norm:.4f}, R2_eff: {r2_eff_norm_full:.4f}")
    print(f"                  [De-normalized] RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return rmse_norm, mae_norm, r2_eff_norm_full, rmse, mae


if __name__ == "__main__":
    # 定义要测试的种子列表
    # seed_list = [
    #             1111, 1222, 1333, 1444, 1555,
    #             1666, 1777, 1888, 1999, 2024,
    #         2025, 2048, 2077, 2099, 2121,
    #         2222, 2333, 2444, 2555, 2666,
    #         2777, 2888, 2999, 3001, 3333,
    #         3456, 3579, 3690, 4040, 5050
    seed_list = [254

    ]

    results = []

    for seed in seed_list:
        # 接收返回的5个全局指标（归一化和未归一化）
        rmse_norm, mae_norm, r2_eff_norm_full, rmse, mae = train_and_evaluate_model(seed=seed)
        results.append((seed, rmse_norm, mae_norm, r2_eff_norm_full, rmse, mae))

    # --- 性能分析与输出 ---

    # 1. 按反归一化 RMSE 升序排序（物理尺度）
    results.sort(key=lambda x: x[4])

    # 拆出最佳结果的5个值
    best_seed, b_rmse_norm, b_mae_norm, b_r2_eff_norm, b_rmse, b_mae = results[0]

    print("\n" + "=" * 120)
    print(f"{'所有种子测试结果 (按 De-normalized RMSE 排序)':^120}")
    print("-" * 120)
    print(
        f"{'种子':<8} | {'RMSE_norm':<12} | {'MAE_norm':<12} | {'R2_eff_norm':<14} | {'RMSE':<12} | {'MAE':<12}")
    print("-" * 120)

    for r in results:
        print(
            f"{r[0]:<8} | {r[1]:<12.4f} | {r[2]:<12.4f} | {r[3]:<14.4f} | {r[4]:<12.4f} | {r[5]:<12.4f}")

    print("-" * 120)
    print(f"🥇 最佳种子 (基于 De-normalized RMSE): {best_seed}")
    print(f"  [Normalized]    RMSE_norm: {b_rmse_norm:.4f}, MAE_norm: {b_mae_norm:.4f}, R2_eff_norm: {b_r2_eff_norm:.4f}")
    print(f"  [De-normalized] RMSE: {b_rmse:.4f}, MAE: {b_mae:.4f}")

    # 2. 平均性能 (Mean ± Std)
    rmses_norm = [r[1] for r in results]
    maes_norm = [r[2] for r in results]
    r2_effs_norm = [r[3] for r in results]
    rmses = [r[4] for r in results]
    maes = [r[5] for r in results]

    print("\n--- 平均性能统计 (所有种子) ---")
    print(f"[Normalized]    RMSE: {np.mean(rmses_norm):.4f} ± {np.std(rmses_norm, ddof=1):.4f}")
    print(f"[Normalized]    MAE:  {np.mean(maes_norm):.4f} ± {np.std(maes_norm, ddof=1):.4f}")
    print(f"[Normalized]    R2_eff: {np.mean(r2_effs_norm):.4f} ± {np.std(r2_effs_norm, ddof=1):.4f}")
    print(f"[De-normalized] RMSE: {np.mean(rmses):.4f} ± {np.std(rmses, ddof=1):.4f}")
    print(f"[De-normalized] MAE:  {np.mean(maes):.4f} ± {np.std(maes, ddof=1):.4f}")
    print("=" * 120)
# import numpy as np
# import torch
# from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

#
# def train_and_evaluate_model(run_id):
#     """
#     不设随机种子
#     每一次调用 = 一次完全独立随机初始化
#     """
#
#     args = {
#         'task_name': 'uvst',
#         'model_id': f'run{run_id}',   # 仅用于区分不同运行
#         'model': 'ConvGRU',
#         'data': 'ssta',
#         'features': 'MS',
#         'learning_rate': 0.0005,
#         'seq_len': 12,
#         'label_len': 12,
#         'pred_len': 12,
#         'd_model': 64,
#         'e_layers': 2,
#         'd_layers': 1,
#         'd_ff': 256,
#         'factor': 1,
#         'embed': 'timeF',
#         'distil': True,
#         'dropout': 0.0,
#         'activation': 'gelu',
#         'use_gpu': True,
#         'train_epochs': 128,
#         'batch_size': 16,
#         'patience': 128,
#         'use_norm': False,
#         'd_core': 512,
#         'freq': 'D',
#         'input_size': 1056,
#         'hidden_size': 64,
#         'output_size': 1,
#         'num_layers': 3,
#         'root_path': r'D:\goole\2025115',
#         'data_path': 'TSuv_data_368.npy',
#         'target_path': r'D:\goole\GOPRdata\Y非nan -1993_2023.xlsx',
#         'target': 'OT',
#         'seasonal_patterns': 'Monthly',
#         'num_workers': 4,
#         'use_amp': False,
#         'output_attention': False,
#         'record_grad': False,
#         'lradj': 'type1',
#         'checkpoints': r'D:\sea level variability\code_nepo\组件消融\东北太平洋组件消融+其他模型\GAconvgru替换为convgru\SOFTS-main\checkpoints',
#         'save_model': True,
#         'device_ids': [0],
#         'scale': True,
#         'num_heads': 4,
#     }
#
#     exp = Exp_Long_Term_Forecast(args)
#
#     print(f"\n>>>>>>> 第 {run_id} 次独立随机运行开始 (ConvGRU) <<<<<<<")
#     exp.train(args)
#     print("训练完成！")
#
#     setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
#         args['task_name'], args['model_id'], args['model'], args['data'], args['features'],
#         args['seq_len'], args['label_len'], args['pred_len'], args['d_model'],
#         args['e_layers'], args['d_layers'], args['d_ff'],
#         args['factor'], args['embed'], args['distil'], args['target']
#     )
#
#     result = exp.test(setting)
#
#     rmse_norm = result['rmse_norm']
#     mae_norm = result['mae_norm']
#     r2_eff_norm = result['r2_eff_norm_full']
#     rmse = result['rmse']
#     mae = result['mae']
#
#     print(
#         f"Run {run_id} 完成 | "
#         f"[Norm] RMSE: {rmse_norm:.4f}, MAE: {mae_norm:.4f}, R2_eff: {r2_eff_norm:.4f} | "
#         f"[Denorm] RMSE: {rmse:.4f}, MAE: {mae:.4f}"
#     )
#
#     return rmse_norm, mae_norm, r2_eff_norm, rmse, mae
#
#
# # =========================
# # 主程序：20 次独立随机运行
# # =========================
# if __name__ == "__main__":
#
#     NUM_RUNS = 40
#     results = []
#
#     for run_id in range(1, NUM_RUNS + 1):
#         metrics = train_and_evaluate_model(run_id)
#         results.append((run_id, *metrics))
#
#     # 拆分指标
#     rmse_norms = [r[1] for r in results]
#     maes_norm = [r[2] for r in results]
#     r2_effs_norm = [r[3] for r in results]
#     rmses = [r[4] for r in results]
#     maes = [r[5] for r in results]
#
#     print("\n" + "=" * 120)
#     print(f"{'ConvGRU：20 次独立随机初始化实验结果':^120}")
#     print("-" * 120)
#     print(f"{'Run':<6} | {'RMSE_norm':<12} | {'MAE_norm':<12} | {'R2_eff_norm':<14} | {'RMSE':<12} | {'MAE':<12}")
#     print("-" * 120)
#
#     for r in results:
#         print(f"{r[0]:<6} | {r[1]:<12.4f} | {r[2]:<12.4f} | {r[3]:<14.4f} | {r[4]:<12.4f} | {r[5]:<12.4f}")
#
#     print("-" * 120)
#
#     print("\n--- 平均性能统计（Mean ± Std，20 次独立运行）---")
#     print(f"[Normalized]    RMSE: {np.mean(rmse_norms):.4f} ± {np.std(rmse_norms, ddof=1):.4f}")
#     print(f"[Normalized]    MAE:  {np.mean(maes_norm):.4f} ± {np.std(maes_norm, ddof=1):.4f}")
#     print(f"[Normalized]    R2_eff: {np.mean(r2_effs_norm):.4f} ± {np.std(r2_effs_norm, ddof=1):.4f}")
#     print(f"[De-normalized] RMSE: {np.mean(rmses):.4f} ± {np.std(rmses, ddof=1):.4f}")
#     print(f"[De-normalized] MAE:  {np.mean(maes):.4f} ± {np.std(maes, ddof=1):.4f}")
#     print("=" * 120)
