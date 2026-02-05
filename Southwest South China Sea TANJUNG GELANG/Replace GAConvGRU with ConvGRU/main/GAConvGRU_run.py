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
    """
    Single run：Training and Evaluation convGRU
    """
    set_seed(seed)

    args = {
        'task_name': 'Convgru',
        'model_id': "convGRU",
        'model': 'convGRU',
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
        'data_path': "anomaly_1993_2018_depth15_filtered.npy",
        'target_path': r"D:\sea level variability\DATA_eio\1589\processed_1589.xlsx",
        'target': "OT",
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        "lradj": "type1",
        'checkpoints': r'D:\sea level variability\code_eio\消融实验\GAconvGRU替换为convgru\SOFTS-main\checkpoints',
        'save_model': True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }

    exp = Exp_Long_Term_Forecast(args)
    print(f"Start training，model ID: {args['model_id']}")

    model = exp._build_model()
    print("Total trainable parameter count：", count_param(model))

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

    exp.train(args)
    print("Training completed!")

    print("开始在Test集上Evaluation...")
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args['task_name'], args['model_id'], args['model'], args['data'], args['features'],
        args['seq_len'], args['label_len'], args['pred_len'], args['d_model'], args['e_layers'],
        args['d_layers'], args['d_ff'], args['factor'], args['embed'], args['distil'], args['target']
    )

    result = exp.test(setting)

    rmse_batch_norm_avg = result['rmse_batch_norm_avg']
    mae_batch_norm_avg = result['mae_batch_norm_avg']
    r2_batch_norm_avg = result['r2_batch_norm_avg']

    rmse_full_norm_avg = result['rmse_full_norm_avg']
    mae_full_norm_avg = result['mae_full_norm_avg']
    r2_full_norm_avg = result['r2_full_norm_avg']
    rmse_full_denorm_avg = result['rmse_full_denorm_avg']
    mae_full_denorm_avg = result['mae_full_denorm_avg']

    print("Evaluation completed！")
    print(f"  BatchNormalization   - RMSE: {rmse_batch_norm_avg:.4f}, MAE: {mae_batch_norm_avg:.4f}, R²_eff: {r2_batch_norm_avg:.4f}")
    print(f"  De-overlappingNormalization   - RMSE: {rmse_full_norm_avg:.4f}, MAE: {mae_full_norm_avg:.4f}, R²_eff: {r2_full_norm_avg:.4f}")
    print(f"  De-overlappingDenormalization - RMSE: {rmse_full_denorm_avg:.4f}, MAE: {mae_full_denorm_avg:.4f}")

    return (
        rmse_batch_norm_avg, mae_batch_norm_avg, r2_batch_norm_avg,
        rmse_full_norm_avg, mae_full_norm_avg, r2_full_norm_avg,
        rmse_full_denorm_avg, mae_full_denorm_avg
    )

if __name__ == "__main__":

    (rmse_batch, mae_batch, r2_batch,
     rmse_norm, mae_norm, r2_norm,
     rmse_denorm, mae_denorm) = train_and_evaluate_model(seed=8240)

    print("\n" + "=" * 90)
    print(f"{'Single run result':^90}")
    print("-" * 90)
    print(f"RMSE(batchN):   {rmse_batch:.4f}")
    print(f"MAE(batchN):    {mae_batch:.4f}")
    print(f"R2(batchN):     {r2_batch:.4f}")
    print("-" * 90)
    print(f"RMSE(norm):     {rmse_norm:.4f}")
    print(f"MAE(norm):      {mae_norm:.4f}")
    print(f"R2_eff(norm):   {r2_norm:.4f}")
    print("-" * 90)
    print(f"RMSE(denorm):   {rmse_denorm:.4f}")
    print(f"MAE(denorm):    {mae_denorm:.4f}")
    print("=" * 90)

