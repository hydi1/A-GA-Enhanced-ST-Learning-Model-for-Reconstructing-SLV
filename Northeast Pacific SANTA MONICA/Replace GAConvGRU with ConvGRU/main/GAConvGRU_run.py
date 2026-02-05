import numpy as np
import random
import torch
from torchinfo import summary
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def train_and_evaluate_model():


    args = {
        'task_name': 'uvst',
        'model_id': "ConvGRU",
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
        'root_path': r'../../Data',
        'data_path': 'TSuv_data_368.npy',
        'target_path': r'../../Data/Y_nonan-1993_2023.xlsx',
        'target': 'OT',
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        'record_grad': False,
        'lradj': 'type1',
        'checkpoints': r'../../checkpoints',
        'save_model': True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }

    exp = Exp_Long_Term_Forecast(args)
    print(f"\n>>>>>>> Start training，model: {args['model']},  <<<<<<<")

    model = exp._build_model()
    print("Total trainable parameter count:", count_param(model))

    # ----- torchinfo summary (real batch) -----
    train_data, train_loader = exp._get_data(flag='train')
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))

    device = torch.device('cuda' if args['use_gpu'] and torch.cuda.is_available() else 'cpu')
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device) if batch_x_mark is not None else None
    batch_y = batch_y.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device) if batch_y_mark is not None else None

    dec_inp = batch_y
    input_data = (batch_x, batch_x_mark, dec_inp, batch_y_mark)
    print(summary(model, input_data=input_data))

    # ----- train -----
    exp.train(args)
    print("Training completed!")

    # ----- test -----
    print("Begin evaluation on the Test set (Denormalization global Metrics after De-overlapping)...")
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args['task_name'], args['model_id'], args['model'], args['data'], args['features'],
        args['seq_len'], args['label_len'], args['pred_len'], args['d_model'], args['e_layers'],
        args['d_layers'], args['d_ff'], args['factor'], args['embed'], args['distil'], args['target']
    )

    result = exp.test(setting)
    rmse_norm = result.get('rmse_norm')
    mae_norm = result.get('mae_norm')
    r2_eff_norm_full = result.get('r2_eff_norm_full')
    rmse = result.get('rmse')
    mae = result.get('mae')

    print(f"Evaluation completed! [Normalized] RMSE: {rmse_norm:.4f}, MAE: {mae_norm:.4f}, R2_eff: {r2_eff_norm_full:.4f}")
    print(f"                    [De-normalized] RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return rmse_norm, mae_norm, r2_eff_norm_full, rmse, mae


if __name__ == "__main__":
    rmse_norm, mae_norm, r2_eff_norm_full, rmse, mae = train_and_evaluate_model()

    print("\n" + "=" * 70)
    print("Single run result")
    print("-" * 70)
    print(f"[Normalized]    RMSE: {rmse_norm:.4f}, MAE: {mae_norm:.4f}, R2_eff: {r2_eff_norm_full:.4f}")
    print(f"[De-normalized] RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    print("=" * 70)
