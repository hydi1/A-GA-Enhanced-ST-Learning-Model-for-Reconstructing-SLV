import numpy as np
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

def count_param(model):
    """Calculate the total trainable parameters of the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate_once():
    """
    单次独立运行：
    - No random seed set
    - no multiple run / seed comparison
    - Each script launch is a completely new experiment
    """

    args = {
        'task_name': 'uvst',
        'model_id': 'single_run',
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
        "use_norm": False,
        'd_core': 512,
        'freq': 'D',
        'input_size': 1056,
        'hidden_size': 64,
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
        'checkpoints': r'D:\project\组件消融\convgru-移除Linear\SOFTS-main\checkpoints',
        "save_model": True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }

    exp = Exp_Long_Term_Forecast(args)

    model = exp._build_model()
    print("Total trainable parameter count：", count_param(model))

    print("\nStart training（单次独立运行）...")
    exp.train(args)
    print("Training completed!")

    print("\n开始在Test集上Evaluation (De-overlapping global Metrics)...")
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

    rmse_norm_full = result.get('rmse_norm_full')
    mae_norm_full = result.get('mae_norm_full')
    r2_eff_norm_full = result.get('r2_eff_norm_full')
    rmse_denorm_full = result.get('rmse_denorm_full')
    mae_denorm_full = result.get('mae_denorm_full')

    print("\n✅ Single runEvaluation结果")
    print(f"[Normalized]    RMSE: {rmse_norm_full:.4f}, MAE: {mae_norm_full:.4f}, R2_eff: {r2_eff_norm_full:.4f}")
    print(f"[De-normalized] RMSE: {rmse_denorm_full:.4f}, MAE: {mae_denorm_full:.4f}")

    return rmse_norm_full, mae_norm_full, r2_eff_norm_full, rmse_denorm_full, mae_denorm_full

if __name__ == "__main__":
    train_and_evaluate_once()
