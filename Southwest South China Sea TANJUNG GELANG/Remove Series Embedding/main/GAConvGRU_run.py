import random
import numpy as np
import torch
from torchinfo import summary
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

def count_param(model):
    return sum(p.numel() for p in model.parameters())

def train_and_evaluate_model(seed=42):

    args = {
        'task_name': 'GAConvgru_610',
        'model_id': "run1",
        'model': 'GAconvgru',
        'data': 'ssta',
        'features': 'MS',
        'learning_rate': 0.0005,
        'seq_len': 12,
        'label_len': 12,
        'pred_len': 12,
        'd_model': 12,
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
        'lradj': "type1",
        'checkpoints': r'D:\sea level variability\code_eio\消融实验\GAconvGRU移除Series Embedding\SOFTS-main\checkpoints',
        'save_model': True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }

    exp = Exp_Long_Term_Forecast(args)
    print(f"Start training | model_id={args['model_id']}")

    model = exp._build_model()
    print("Total parameter count：", count_param(model))

    train_data, train_loader = exp._get_data(flag='train')
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))

    device = torch.device('cuda' if args['use_gpu'] and torch.cuda.is_available() else 'cpu')
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device) if batch_x_mark is not None else None
    batch_y_mark = batch_y_mark.float().to(device) if batch_y_mark is not None else None

    dec_inp = batch_y
    print(summary(model, input_data=(batch_x, batch_x_mark, dec_inp, batch_y_mark)))

    exp.train(args)
    print("Training completed")

    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args['task_name'], args['model_id'], args['model'], args['data'], args['features'],
        args['seq_len'], args['label_len'], args['pred_len'], args['d_model'], args['e_layers'],
        args['d_layers'], args['d_ff'], args['factor'], args['embed'], args['distil'], args['target']
    )

    result = exp.test(setting)

    print("\n===== Single runEvaluation结果 =====")
    print(
        f"Batch-Norm : RMSE={result['rmse_batch_norm_avg']:.4f}, "
        f"MAE={result['mae_batch_norm_avg']:.4f}, "
        f"R2={result['r2_batch_norm_avg']:.4f}\n"
        f"Deoverlap-Norm : RMSE={result['rmse_full_norm_avg']:.4f}, "
        f"MAE={result['mae_full_norm_avg']:.4f}, "
        f"R2_eff={result['r2_full_norm_avg']:.4f}\n"
        f"Deoverlap-Denorm : RMSE={result['rmse_full_denorm_avg']:.4f}, "
        f"MAE={result['mae_full_denorm_avg']:.4f}"
    )

    return result

if __name__ == "__main__":

    _ = train_and_evaluate_model()
