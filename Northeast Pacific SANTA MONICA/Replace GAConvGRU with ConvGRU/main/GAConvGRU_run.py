import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from torchinfo import summary
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def set_seed(seed):
    """
    Set random seed to ensure reproducibility
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
    Train and Evaluate SOFTS model using provided Training, Validation and Test data
    """

    set_seed(seed)

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

    exp = Exp_Long_Term_Forecast(args)
    print(f"\n>>>>>>> Start training，模型: {args['model']}, 种子: {seed} <<<<<<<")
    model = exp._build_model()

    print("Total trainable parameter count：", count_param(model))

    train_data, train_loader = exp._get_data(flag='train')
    batch = next(iter(train_loader))
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch

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

    print("开始在Test集上Evaluation (Denormalization global Metrics after De-overlapping)...")
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

    print(f"种子 {seed} Evaluation completed！[Normalized] RMSE: {rmse_norm:.4f}, MAE: {mae_norm:.4f}, R2_eff: {r2_eff_norm_full:.4f}")
    print(f"                  [De-normalized] RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return rmse_norm, mae_norm, r2_eff_norm_full, rmse, mae

if __name__ == "__main__":

    seed_list = [254

    ]

    results = []

    for seed in seed_list:

        rmse_norm, mae_norm, r2_eff_norm_full, rmse, mae = train_and_evaluate_model(seed=seed)
        results.append((seed, rmse_norm, mae_norm, r2_eff_norm_full, rmse, mae))

    results.sort(key=lambda x: x[4])

    best_seed, b_rmse_norm, b_mae_norm, b_r2_eff_norm, b_rmse, b_mae = results[0]

    print("\n" + "=" * 120)
    print(f"{'所有种子Test结果 (按 De-normalized RMSE 排序)':^120}")
    print("-" * 120)
    print(
        f"{'种子':<8} | {'RMSE_norm':<12} | {'MAE_norm':<12} | {'R2_eff_norm':<14} | {'RMSE':<12} | {'MAE':<12}")
    print("-" * 120)

    for r in results:
        print(
            f"{r[0]:<8} | {r[1]:<12.4f} | {r[2]:<12.4f} | {r[3]:<14.4f} | {r[4]:<12.4f} | {r[5]:<12.4f}")

    print("-" * 120)
    print(f"🥇 Best seed (based on De-normalized RMSE): {best_seed}")
    print(f"  [Normalized]    RMSE_norm: {b_rmse_norm:.4f}, MAE_norm: {b_mae_norm:.4f}, R2_eff_norm: {b_r2_eff_norm:.4f}")
    print(f"  [De-normalized] RMSE: {b_rmse:.4f}, MAE: {b_mae:.4f}")

    rmses_norm = [r[1] for r in results]
    maes_norm = [r[2] for r in results]
    r2_effs_norm = [r[3] for r in results]
    rmses = [r[4] for r in results]
    maes = [r[5] for r in results]

    print("\n--- Average performance statistics (all seeds) ---")
    print(f"[Normalized]    RMSE: {np.mean(rmses_norm):.4f} ± {np.std(rmses_norm, ddof=1):.4f}")
    print(f"[Normalized]    MAE:  {np.mean(maes_norm):.4f} ± {np.std(maes_norm, ddof=1):.4f}")
    print(f"[Normalized]    R2_eff: {np.mean(r2_effs_norm):.4f} ± {np.std(r2_effs_norm, ddof=1):.4f}")
    print(f"[De-normalized] RMSE: {np.mean(rmses):.4f} ± {np.std(rmses, ddof=1):.4f}")
    print(f"[De-normalized] MAE:  {np.mean(maes):.4f} ± {np.std(maes, ddof=1):.4f}")
    print("=" * 120)
