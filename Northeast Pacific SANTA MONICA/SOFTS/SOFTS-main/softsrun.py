import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

# def train_and_evaluate_model():
#     """
#     Train and evaluate the SOFTS model using the provided training, validation, and test data.
#     """
#     # Configure arguments for the experiment
#     args = {
#         'task_name': 'TSdepth51993-2023',
#         'model_id': 'train',
#         'model': 'SOFTS',
#         'data': 'ssta',
#         'features': 'MS',
#         'learning_rate': 0.0001,
#         'seq_len': 12,
#         'label_len': 12,
#         'pred_len': 12,
#         'd_model': 64,
#         'e_layers': 2,
#         'd_layers': 1,
#         'd_ff': 256,
#         'factor': 1,
#         'embed': 'timeF',
#         'distil': True,
#         'dropout': 0.0,
#         'activation': 'gelu',
#         'use_gpu': True,
#         'train_epochs': 128,
#         'batch_size': 16,
#         'patience': 128,
#         # 'loss': 'MSE',
#         "use_norm": False,
#         'd_core': 512,
#         'freq': 'D',
#         # 'enc_in': 441,
#         # 'dec_in': 441,
#         # 'c_out': 441,
#         'root_path': r'D:\goole\2025115',
#         "data_path": 'TSuv_data_368.npy',
#         "target_path": r"D:\goole\GOPRdata\Y非nan -1993_2023.xlsx " ,
#         'target': "OT",  # OT 是什么意思
#         'seasonal_patterns': 'Monthly',
#         'num_workers': 4,
#         'use_amp': False,
#         'output_attention':False,
#         "lradj": "type1",
#         # 'learning_rate': 0.0001,
#         'checkpoints': r'D:\project\SOFTS_TS -反归一化\SOFTS-main\checkpoints',
#         "save_model":True,
#         'device_ids':[0],
#         'scale': True,
#
#     }
#
#     # Initialize experiment
#     exp = Exp_Long_Term_Forecast(args)
#     # Start training
#     print(f"Starting training with model ID: {args['model_id']}")
#     exp.train(args)
#     print("Training completed!")
#
#     # Evaluate on validation data
#     print("Starting evaluation on validation set...")
#     setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
#         args['task_name'],
#         args['model_id'],
#         args['model'],
#         args['data'],
#         args['features'],
#         args['seq_len'],
#         args['label_len'],
#         args['pred_len'],
#         args['d_model'],
#         args['e_layers'],
#         args['d_layers'],
#         args['d_ff'],
#         args['factor'],
#         args['embed'],
#         args['distil'],
#         args['target']
#     )
#     exp.test(setting)
#     print("Evaluation completed!")
#
#     # # Train the model
#     # print(f"Starting training with model ID: {args['model_id']}")
#     # exp.train(args, x_train, x_val, y_train, y_val)  # assuming train() takes data as arguments
#     # print("Training completed!")
#     #
#     # # Evaluate on validation data
#     # print("Starting evaluation on validation set...")
#     # exp.test(x_val, y_val)  # assuming test() takes validation data as arguments
#     # print("Validation evaluation completed!")
#     #
#     # # Evaluate on test data
#     # print("Starting evaluation on test set...")
#     # exp.test(x_test, y_test)  # assuming test() takes test data as arguments
#     # print("Test evaluation completed!")
#
# if __name__ == "__main__":
#     # Start training and evaluation process
#     train_and_evaluate_model()
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.model_selection import train_test_split
# from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
#
# def train_and_evaluate_model():
#     """
#     Train and evaluate the SOFTS model using the provided training, validation, and test data.
#     """
#     # Configure arguments for the experiment
#     args = {
#         'task_name': 'TSdepth51993-2023',
#         'model_id': 'train',
#         'model': 'SOFTS',
#         'data': 'ssta',
#         'features': 'MS',
#         'learning_rate': 0.0001,
#         'seq_len': 12,
#         'label_len': 12,
#         'pred_len': 12,
#         'd_model': 64,
#         'e_layers': 2,
#         'd_layers': 1,
#         'd_ff': 256,
#         'factor': 1,
#         'embed': 'timeF',
#         'distil': True,
#         'dropout': 0.0,
#         'activation': 'gelu',
#         'use_gpu': True,
#         'train_epochs': 128,
#         'batch_size': 16,
#         'patience': 128,
#         # 'loss': 'MSE',
#         "use_norm": False,
#         'd_core': 512,
#         'freq': 'D',
#         # 'enc_in': 441,
#         # 'dec_in': 441,
#         # 'c_out': 441,
#         'root_path': r'D:\goole\2025115',
#         "data_path": 'TSuv_data_368.npy',
#         "target_path": r"D:\goole\GOPRdata\Y非nan -1993_2023.xlsx " ,
#         'target': "OT",  # OT 是什么意思
#         'seasonal_patterns': 'Monthly',
#         'num_workers': 4,
#         'use_amp': False,
#         'output_attention':False,
#         "lradj": "type1",
#         # 'learning_rate': 0.0001,
#         'checkpoints': r'D:\project\SOFTS_TS -反归一化\SOFTS-main\checkpoints',
#         "save_model":True,
#         'device_ids':[0],
#         'scale': True,
#
#     }
#
#     # Initialize experiment
#     exp = Exp_Long_Term_Forecast(args)
#     # Start training
#     print(f"Starting training with model ID: {args['model_id']}")
#     exp.train(args)
#     print("Training completed!")
#
#     # Evaluate on validation data
#     print("Starting evaluation on validation set...")
#     setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
#         args['task_name'],
#         args['model_id'],
#         args['model'],
#         args['data'],
#         args['features'],
#         args['seq_len'],
#         args['label_len'],
#         args['pred_len'],
#         args['d_model'],
#         args['e_layers'],
#         args['d_layers'],
#         args['d_ff'],
#         args['factor'],
#         args['embed'],
#         args['distil'],
#         args['target']
#     )
#     exp.test(setting)
#     print("Evaluation completed!")
#
#     # # Train the model
#     # print(f"Starting training with model ID: {args['model_id']}")
#     # exp.train(args, x_train, x_val, y_train, y_val)  # assuming train() takes data as arguments
#     # print("Training completed!")
#     #
#     # # Evaluate on validation data
#     # print("Starting evaluation on validation set...")
#     # exp.test(x_val, y_val)  # assuming test() takes validation data as arguments
#     # print("Validation evaluation completed!")
#     #
#     # # Evaluate on test data
#     # print("Starting evaluation on test set...")
#     # exp.test(x_test, y_test)  # assuming test() takes test data as arguments
#     # print("Test evaluation completed!")
#
# if __name__ == "__main__":
#     # Start training and evaluation process
#     train_and_evaluate_model()
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchinfo import summary
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

# 定义计算参数量的函数
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def set_seed(seed):
    """
    设置随机种子以确保可重复性
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
    Train and evaluate the SOFTS model using the provided training, validation, and test data.
    """
    set_seed(seed)



    args = {
        'task_name': 'TSdepth51993-2023',
        'model_id': 200,
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
        "use_norm": False,
        'd_core': 512,
        'freq': 'D',
        'root_path': r'D:\海平面变率\nepo',
        'data_path': 'Suvz_so_uovo_data_368_smart_filled.npy',
        "target_path": r"D:\goole\GOPRdata\Y非nan -1993_2023.xlsx " ,
        'target': "OT",
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention':False,
        "lradj": "type1",
        'checkpoints': r'D:\sea level variability\code_nepo\SOFTS_TS -12\SOFTS-main\checkpoints',
        "save_model":True,
        'device_ids':[0],
        'scale': True,

        }
    # 初始化实验
    exp = Exp_Long_Term_Forecast(args)
    print(f"开始训练，模型 ID: {args['model_id']}, 种子: {seed}")

    # 显式构建模型以访问它
    model = exp._build_model()

    # 计算并打印参数量
    print("总可训练参数量：", count_param(model))

    # 获取一个真实的输入批次
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

    exp.train(args)
    print("训练完成！")

    # 在验证集上评估
    print("开始在验证集上评估...")
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args['task_name'], args['model_id'], args['model'], args['data'], args['features'],
        args['seq_len'], args['label_len'], args['pred_len'], args['d_model'], args['e_layers'],
        args['d_layers'], args['d_ff'], args['factor'], args['embed'], args['distil'], args['target']
    )
    result = exp.test(setting)
    rmse_batch_unnorm_avg = result['Full Normalized: rmse']
    mae_batch_unnorm_avg = result['Full Normalized: mae']
    print(f"种子 {seed} 评估完成！RMSE: {rmse_batch_unnorm_avg:.3f}, MAE: {mae_batch_unnorm_avg:.3f}")
    return rmse_batch_unnorm_avg, mae_batch_unnorm_avg


if __name__ == "__main__":
    # 测试多个种子
    seed_list = [1888]  # 定义要测试的种子列表
    results = []

    for seed in seed_list:
        rmse, mae = train_and_evaluate_model(seed=seed)
        results.append((seed, rmse, mae))

    # 按 RMSE 排序，找到最好的种子
    results.sort(key=lambda x: x[1])  # 按 RMSE 升序排序
    best_seed, best_rmse, best_mae = results[0]
    print("\n所有种子测试结果：")
    for seed, rmse, mae in results:
        print(f"种子: {seed}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    print(f"\n最佳种子: {best_seed}, 最佳 RMSE: {best_rmse:.3f}, 最佳 MAE: {best_mae:.3f}")

    # 计算平均性能
    rmses = [r[1] for r in results]
    maes = [r[2] for r in results]
    mean_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)
    mean_mae = np.mean(maes)
    std_mae = np.std(maes)
    print(f"\n平均 RMSE: {mean_rmse:.3f} ± {std_rmse:.3f}")
    print(f"平均 MAE: {mean_mae:.3f} ± {std_mae:.3f}")