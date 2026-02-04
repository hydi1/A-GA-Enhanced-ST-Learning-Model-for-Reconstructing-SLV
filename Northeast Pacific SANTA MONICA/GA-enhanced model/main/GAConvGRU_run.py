import numpy as np
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from torchinfo import summary


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


def train_and_evaluate_model():
    """
    单次独立运行：训练并评估模型（不固定随机种子）
    """

    # 配置实验参数
    args = {
        'task_name': 'uvst',
        'model_id': "run1",
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
        'use_norm': False,
        'd_core': 512,
        'freq': 'D',
        'input_size': 1056,
        'hidden_size': 1056,
        'output_size': 1,
        'num_layers': 3,
        'root_path': r'D:\CGI2025代码\Northeast Pacific SANTA MONICA\Data',
        'data_path': 'TSuv_data_368.npy',
        'target_path': r"D:\CGI2025代码\Northeast Pacific SANTA MONICA\Data\Y非nan -1993_2023.xlsx",
        'target': 'OT',
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        'lradj': 'type1',
        'checkpoints': CHECKPOINT_DIR,
        'save_model': True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }

    # 初始化实验
    exp = Exp_Long_Term_Forecast(args)

    # 开始训练
    print(f"开始训练，模型 ID: {args['model_id']}")
    model = exp._build_model()

    # 计算并打印参数量
    print("总可训练参数量：", count_param(model))

    # 获取一个真实的输入批次（用于 summary）
    train_data, train_loader = exp._get_data(flag='train')
    batch = next(iter(train_loader))
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch

    # 移动到与模型相同的设备
    device = torch.device('cuda' if args['use_gpu'] and torch.cuda.is_available() else 'cpu')
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device) if batch_x_mark is not None else None
    batch_y = batch_y.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device) if batch_y_mark is not None else None

    # 构造 x_dec（根据你的 forward 方法逻辑）
    dec_inp = batch_y

    # 使用 input_data 传递实际输入
    input_data = (batch_x, batch_x_mark, dec_inp, batch_y_mark)
    print(summary(model, input_data=input_data))

    # 训练
    exp.train(args)
    print("训练完成！")

    # 评估
    print("开始在验证集上评估...")
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

    # 提取指标（根据简化后的 test 函数）
    rmse_norm = result.get('rmse_norm')
    mae_norm = result.get('mae_norm')
    r2_eff_norm = result.get('r2_eff_norm')
    rmse_unnorm = result.get('rmse_unnorm')
    mae_unnorm = result.get('mae_unnorm')

    print("评估完成！")
    print(f"  归一化+去重叠: RMSE={rmse_norm:.4f}, MAE={mae_norm:.4f}, R2_eff={r2_eff_norm:.4f}")
    print(f"  反归一化+去重叠: RMSE={rmse_unnorm:.4f}, MAE={mae_unnorm:.4f}")

    return rmse_norm, mae_norm, r2_eff_norm, rmse_unnorm, mae_unnorm


if __name__ == "__main__":
    rmse_norm, mae_norm, r2_eff_norm, rmse_unnorm, mae_unnorm = train_and_evaluate_model()

    print("\n" + "=" * 60)
    print(f"{'单次运行结果':^60}")
    print("-" * 60)
    print(f"归一化+去重叠 RMSE: {rmse_norm:.4f}")
    print(f"归一化+去重叠 MAE:  {mae_norm:.4f}")
    print(f"R2_eff(归一化+去重叠): {r2_eff_norm:.4f}")
    print(f"反归一化+去重叠 RMSE: {rmse_unnorm:.4f}")
    print(f"反归一化+去重叠 MAE:  {mae_unnorm:.4f}")
    print("=" * 60)

