import numpy as np
import torch
from torchinfo import summary
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

def count_param(model):
    return sum(p.numel() for p in model.parameters())

def train_and_evaluate_model():

    args = {
        'task_name': 'TSdepth51993-2023',
        'model_id': 'run1',
        'model': 'SOFTS',
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

        'root_path': r'../../Data',
        'data_path': 'Anomalies_2004-2022_filtered.npy',
        'target_path': r"../../Data/4processed_HIERRO_nomiss.xlsx",
        'target': 'OT',

        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        'lradj': 'type1',
        'checkpoints': r'../../checkpoints',
        'save_model': True,
        'device_ids': [0],
        'scale': True,
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

    print("Start Evaluation on Validation set...")
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args['task_name'], args['model_id'], args['model'], args['data'], args['features'],
        args['seq_len'], args['label_len'], args['pred_len'], args['d_model'], args['e_layers'],
        args['d_layers'], args['d_ff'], args['factor'], args['embed'], args['distil'], args['target']
    )

    result = exp.test(setting)
    rmse = result['Full Normalized: rmse']
    mae = result['Full Normalized: mae']

    print("Evaluation completed！")
    print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}")

    return rmse, mae

if __name__ == "__main__":
    rmse, mae = train_and_evaluate_model()

    print("\n" + "=" * 50)
    print(f"{'Single run result':^50}")
    print("-" * 50)
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print("=" * 50)

