import numpy as np
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def count_param(model):
    """计算模型的可训练参数总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_evaluate_once():
    """
    单次训练 + 单次测试（不设置随机种子，每次运行独立）
    """

    args = {
        'task_name': 'vst',
        'model_id': "single_run",
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
        "use_norm": False,
        'd_core': 512,
        'freq': 'D',
        'input_size': 1056,
        'hidden_size': 1056,
        'output_size': 1,
        'num_layers': 3,
        'root_path': r'D:\goole\2025115',
        "data_path": 'TSuv_data_368.npy',
        "target_path": r"D:\goole\GOPRdata\Y非nan -1993_2023.xlsx",
        'target': "OT",
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        "lradj": "type1",
        'checkpoints': r'D:\project\组件消融\convgru-移除path1\SOFTS-main\checkpoints',
        "save_model": True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }

    exp = Exp_Long_Term_Forecast(args)
    print(f"开始训练，模型 ID: {args['model_id']}（单次独立运行，不固定随机种子）")

    model = exp._build_model()
    print("总可训练参数量：", count_param(model))

    exp.train(args)
    print("训练完成！")

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

    result = exp.test(setting)
    rmse_norm = result.get('rmse_norm', None)
    mae_norm = result.get('mae_norm', None)
    r2_eff_norm_full = result.get('r2_eff_norm_full', None)
    rmse = result.get('rmse', None)
    mae = result.get('mae', None)

    print(
        f"评估完成！RMSE(norm): {rmse_norm:.4f}, MAE(norm): {mae_norm:.4f}, "
        f"R2_eff(norm,full): {r2_eff_norm_full:.4f} | RMSE: {rmse:.4f}, MAE: {mae:.4f}"
    )
    return rmse_norm, mae_norm, r2_eff_norm_full, rmse, mae


if __name__ == "__main__":
    train_and_evaluate_once()
