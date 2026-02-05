limport numpy as np
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

def count_param(model):
    """Calculate the total number of trainable parameters of the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate_once():
    """
    Single Training + Single Test (no random seed set, each run is independent)
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
        'root_path': r'../../Data',
        "data_path": 'TSuv_data_368.npy',
        "target_path": r"../../Data/Y_nonan-1993_2023.xlsx",
        'target': "OT",
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        "lradj": "type1",
        'checkpoints': r'../../checkpoints',
        "save_model": True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }

    exp = Exp_Long_Term_Forecast(args)
    print(f"Start training，model ID: {args['model_id']}")

    model = exp._build_model()
    print("Total trainable parameter count：", count_param(model))

    exp.train(args)
    print("Training completed!")

    print("Begin evaluation on the Test set (De-overlapping global Metrics)...")
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
        f"Evaluation completed！RMSE(norm): {rmse_norm:.4f}, MAE(norm): {mae_norm:.4f}, "
        f"R2_eff(norm,full): {r2_eff_norm_full:.4f} | RMSE: {rmse:.4f}, MAE: {mae:.4f}"
    )
    return rmse_norm, mae_norm, r2_eff_norm_full, rmse, mae

if __name__ == "__main__":
    train_and_evaluate_once()

