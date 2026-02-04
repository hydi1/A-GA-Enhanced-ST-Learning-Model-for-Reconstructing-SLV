import numpy as np
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


# 定义计算参数量的函数
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def train_and_evaluate_model():
    """
    单次独立运行 GAconvgru（不固定随机种子）
    """

    # 配置实验参数
    args = {
        'task_name': 'GAConvgru_610',
        'model_id': 'run1',
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
        'hidden_size': 16,
        'output_size': 1,
        'num_layers': 3,

        'root_path': r'D:\sea level variability\DATA_eio\1589',
        'data_path': "anomaly_1993_2018_depth15_filtered.npy",  # (306, 4, 15, 11, 15)
        'target_path': r"D:\sea level variability\DATA_eio\1589\processed_1589.xlsx",
        'target': "OT",

        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        'lradj': "type1",
        'checkpoints': r'D:\sea level variability\code_eio\消融实验\GAconvGRU移除Linear\SOFTS-main\checkpoints',
        'save_model': True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }

    # 初始化实验
    exp = Exp_Long_Term_Forecast(args)
    print(f"开始训练，模型 ID: {args['model_id']}")

    model = exp._build_model()
    print("总可训练参数量：", count_param(model))

    # ===================== 训练 =====================
    exp.train(args)
    print("训练完成！")

    # ===================== 评估 =====================
    print("开始在验证集上评估...")
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args['task_name'], args['model_id'], args['model'], args['data'], args['features'],
        args['seq_len'], args['label_len'], args['pred_len'], args['d_model'], args['e_layers'],
        args['d_layers'], args['d_ff'], args['factor'], args['embed'], args['distil'], args['target']
    )

    result = exp.test(setting)

    # ===== batch（归一化）3 指标 =====
    rmse_batch_norm_avg = result['rmse_batch_norm_avg']
    mae_batch_norm_avg = result['mae_batch_norm_avg']
    r2_batch_norm_avg = result['r2_batch_norm_avg']

    # ===== 去重叠 5 指标 =====
    rmse_full_norm_avg = result['rmse_full_norm_avg']
    mae_full_norm_avg = result['mae_full_norm_avg']
    r2_full_norm_avg = result['r2_full_norm_avg']
    rmse_full_denorm_avg = result['rmse_full_denorm_avg']
    mae_full_denorm_avg = result['mae_full_denorm_avg']

    print("评估完成！")
    print(
        f"BatchN: RMSE={rmse_batch_norm_avg:.3f}, MAE={mae_batch_norm_avg:.3f}, R2_eff={r2_batch_norm_avg:.4f}\n"
        f"Deoverlap Norm: RMSE={rmse_full_norm_avg:.3f}, MAE={mae_full_norm_avg:.3f}, R2_eff={r2_full_norm_avg:.4f}\n"
        f"Deoverlap Denorm: RMSE={rmse_full_denorm_avg:.3f}, MAE={mae_full_denorm_avg:.3f}"
    )

    return (
        rmse_batch_norm_avg,
        mae_batch_norm_avg,
        r2_batch_norm_avg,
        rmse_full_norm_avg,
        mae_full_norm_avg,
        r2_full_norm_avg,
        rmse_full_denorm_avg,
        mae_full_denorm_avg
    )


if __name__ == "__main__":
    (rmse_b, mae_b, r2_b,
     rmse_n, mae_n, r2_n,
     rmse_d, mae_d) = train_and_evaluate_model()

    print("\n" + "=" * 90)
    print(f"{'单次运行结果（Batch归一化 + 去重叠）':^90}")
    print("-" * 90)
    print(f"RMSE(batchN):   {rmse_b:.4f}")
    print(f"MAE(batchN):    {mae_b:.4f}")
    print(f"R2(batchN):     {r2_b:.4f}")
    print("-" * 90)
    print(f"RMSE(norm):     {rmse_n:.4f}")
    print(f"MAE(norm):      {mae_n:.4f}")
    print(f"R2_eff(norm):   {r2_n:.4f}")
    print("-" * 90)
    print(f"RMSE(denorm):   {rmse_d:.4f}")
    print(f"MAE(denorm):    {mae_d:.4f}")
    print("=" * 90)
