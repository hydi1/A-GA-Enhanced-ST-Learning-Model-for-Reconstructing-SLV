
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
    使用提供的训练、验证和测试数据训练并评估 GAconvgru 模型
    """
    # 设置随机种子
    set_seed(seed)

    # 配置实验参数
    args = {
        'task_name': 'Transformerdepth41993-2023',
        'model_id': f"seed{seed}",
        'model': 'Transformer',
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
        # 'loss': 'MSE',
        "use_norm": False,
        'd_core': 512,
        'freq': 'D',
        'input_size': 4 * 29 * 6 * 11,
        'hidden_size': 64,
        'output_size': 1,  # 假设是回归任务，输出一个预测值
        'num_layers': 3,  # 使用2层GRU
        'root_path': r'D:\goole\2025115',
        "data_path": 'TSuv_data_368.npy',
        "target_path": r"D:\goole\GOPRdata\Y非nan -1993_2023.xlsx",
        'target': "OT",  # OT 是什么意思
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        "lradj": "type1",
        # 'learning_rate': 0.0001,
        'checkpoints': r'D:\sea level variability\code_nepo\GRU - 12\SOFTS-main\checkpoints',
        "save_model": True,
        'device_ids': [0],
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
    rmse_batch_unnorm_avg = result['rmse_full_norm_avg']
    mae_batch_unnorm_avg = result['mae_full_norm_avg']
    print(f"种子 {seed} 评估完成！RMSE: {rmse_batch_unnorm_avg:.3f}, MAE: {mae_batch_unnorm_avg:.3f}")
    return rmse_batch_unnorm_avg, mae_batch_unnorm_avg

if __name__ == "__main__":
    # 测试多个种子
    seed_list = [42, 43, 44, 45, 46,
        100, 200, 300, 400, 500,
        123, 234, 345, 456, 567,
        678, 789, 888, 999, 1000,

        1666, 1777, 1888, 1999, 2024,
        2025, 2048, 2077, 2099, 2121,
        2222, 2333, 2444, 2555, 2666,
        2777, 2888, 2999, 3001, 3333,
        3456, 3579, 3690, 4040, 5050] 
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
