import numpy as np
import torch
from torchinfo import summary
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

def count_param(model):
    return sum(p.numel() for p in model.parameters())

def train_and_evaluate_model():

    args = {
        'task_name': 'uvst',
        'model_id': 'run1',
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
        'use_norm': False,
        'd_core': 512,
        'freq': 'D',
        'input_size': 720,
        'hidden_size': 64,
        'output_size': 1,
        'num_layers': 3,
        'root_path': r'D:\sea level variability\DATA_neao',
        'data_path': 'Anomalies_2004-2022_filtered_reordered.npy',
        'target_path': r"D:\sea level variability\DATA_neao\4processed_HIERRO_nomiss.xlsx",
        'target': 'OT',
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        'lradj': 'type1',
        'checkpoints': r'D:\sea level variability\code_neao\消融实验\GAconvGRU替换为convgru\SOFTS-main\checkpoints',
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
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device) if batch_x_mark is not None else None
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

    rmse_norm = result['rmse_full_norm_avg']
    mae_norm = result['mae_full_norm_avg']
    r2_norm = result['r2_full_norm_avg']
    rmse_denorm = result['rmse_full_denorm_avg']
    mae_denorm = result['mae_full_denorm_avg']

    print("Evaluation completed！")
    print(f"NormalizationDe-overlapping: RMSE={rmse_norm:.4f}, MAE={mae_norm:.4f}, R²_eff={r2_norm:.4f}")
    print(f"DenormalizationDe-overlapping: RMSE={rmse_denorm:.4f}, MAE={mae_denorm:.4f}")

    return rmse_norm, mae_norm, r2_norm, rmse_denorm, mae_denorm

if __name__ == "__main__":
    rmse_n, mae_n, r2_n, rmse_d, mae_d = train_and_evaluate_model()

    print("\n" + "=" * 70)
    print(f"{'Single run Result':^70}")
    print("-" * 70)
    print(f"NormalizationDe-overlapping RMSE: {rmse_n:.4f}")
    print(f"NormalizationDe-overlapping MAE : {mae_n:.4f}")
    print(f"NormalizationDe-overlapping R²_eff: {r2_n:.4f}")
    print(f"DenormalizationDe-overlapping RMSE: {rmse_d:.4f}")
    print(f"DenormalizationDe-overlapping MAE : {mae_d:.4f}")
    print("=" * 70)


