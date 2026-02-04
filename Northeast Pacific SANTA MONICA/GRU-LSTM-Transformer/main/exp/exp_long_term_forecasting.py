from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Npy
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        return torch.sqrt(self.mse_loss(outputs, targets))
class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)



    def _build_model(self):
        
        model = self.model_dict[self.args['model']].Model(self.args).float()
        if self.args['use_gpu']:
            model = nn.DataParallel(model, device_ids=self.args['device_ids'])
        return model
    #data_provider(self.args, flag) 是一个外部数据处理函数
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        #数据集对象data_set ,数据加载器data_loader
        return data_set, data_loader
    #创建Adam优化器
    def _select_optimizer(self):
        #torch.optim.Adam 初始化优化器  self.model.parameters() 获取模型的所有可训练参数
        model_optim = optim.Adam(self.model.parameters(), lr=self.args['learning_rate'],eps=1e-4)
        # print("学习率是是",self.args['learning_rate'])
        return model_optim
    #为模型选择损失函数
    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion
    #用于验证模型性能的函数 它通过遍历验证数据集计算模型的平均损失（total_loss),用于衡量模型在验证集上的表现
    def vali(self, vali_data, vali_loader, criterion):
        #AverageMeter() 用于计算平均损失
        total_loss = AverageMeter()
        #设置模型为评估模式
        self.model.eval()
        #禁用梯度计算，减少现存占用和计算开销
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                #batch_x:输入时间序列，形状[B,L，D],被转换为浮点数并移动到指定设备
                batch_x = batch_x.float().to(self.device)
                # print("vali_batch_x.shape", batch_x.shape)#torch.Size([15, 96, 441])?????为甚是15？
                #batch_y:目标时间序列，形状【B,L,D]
                batch_y = batch_y.float()
                # print("vali_batch_y.shape", batch_y.shape)#torch.Size([15, 96, 1])

                if 'PEMS' in self.args['data'] or 'Solar' in self.args['data']:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = batch_y.float().to(self.device)
                if self.args['use_amp']:
                    with torch.cuda.amp.autocast():
                        #output_attention是否在编码中输出注意力
                        if self.args['output_attention']:
                            #调用 self.model，输入历史序列和解码器的初始输入，输出预测值 outputs
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args['output_attention']:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args['features'] == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.update(loss.item(), batch_x.size(0))
        total_loss = total_loss.avg
        self.model.train()
        return total_loss

    def train(self, setting):
        #调用 self._get_data 方法，将输入数据（训练、验证、测试）封装成 Dataset 和 DataLoader
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        first_13_values = list(setting.values())[:13]  # 取前13个值
        result_string = '_'.join(map(str, first_13_values))
        path = os.path.join(self.args['checkpoints'], result_string)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        print("共{}个训练批次".format(train_steps))
        early_stopping = EarlyStopping(patience=self.args['patience'], verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args['use_amp']:
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args['train_epochs']):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args['data'] or 'Solar' in self.args['data']:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = batch_y.float().to(self.device)
                # encoder - decoder
                if self.args['use_amp']:
                    with torch.cuda.amp.autocast():
                        if self.args['output_attention']:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args['features'] == 'MS' else 0
                        outputs = outputs[:, -self.args['pred_len']:, f_dim:]
                        batch_y = batch_y[:, -self.args['pred_len']:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)

                else:
                    if self.args['output_attention']:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # print("outputs.shape", outputs.shape)#torch.Size([32, 96, 441])
                    f_dim = -1 if self.args['features'] == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                loss_float = loss.item()
                train_loss.append(loss_float)
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_float))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.args['train_epochs'] - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

                if self.args['use_amp']:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            #根据当前的训练轮次调整优化器的学习率，model_optim代表模型的优化器对象，epoch+1代表当前轮次的索引，self.args代表学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        #拼接出完整的保存路径
        best_model_path = path + '/' + 'checkpoint.pth'
        #torch.load(best_model_path)：加载保存在 checkpoint.pth 文件中的模型权重参数
        # self.model.load_state_dict(...)：将加载的参数赋给当前模型 self.model，用于恢复模型的状态。
        self.model.load_state_dict(torch.load(best_model_path))
        if not self.args['save_model']:
            import shutil
            shutil.rmtree(path)
        return self.model

    # 假设这些类已在其他地方定义
    class RMSELoss(nn.Module):
        def __init__(self):
            super(RMSELoss, self).__init__()

        def forward(self, pred, target):
            return torch.sqrt(nn.MSELoss()(pred, target))

    class AverageMeter(object):
        """Computes and stores the average and current value"""

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    # 假设 self 所在的类（如 Exp_Main）中包含了 _get_data, model, device, args 等属性

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # 定义损失函数
        rmse_loss = RMSELoss()
        mae_loss = nn.L1Loss()

        # --- 1. 基于 batch 的损失 (Batch-based Losses) ---
        # 损失在模型的归一化空间上 (Normalized, 模型尺度)
        rmse_batch_norm = AverageMeter()
        mae_batch_norm = AverageMeter()
        # 损失在数据的原始空间上 (Unnormalized, 原始尺度)
        rmse_batch_unnorm = AverageMeter()
        mae_batch_unnorm = AverageMeter()

        # --- 2. 基于去重叠后的损失 (Full Sequence Losses) ---
        # 损失在模型的归一化空间上 (Normalized, 模型尺度)
        rmse_full_norm = AverageMeter()
        mae_full_norm = AverageMeter()
        # 损失在数据的原始空间上 (Unnormalized, 原始尺度)
        rmse_full_unnorm = AverageMeter()
        mae_full_unnorm = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            all_outputs_norm = []  # 保存所有预测结果 (归一化, 用于去重叠)
            all_batch_y_norm = []  # 保存所有真实值 (归一化, 用于去重叠)

            # 预定义测试集的完整长度 (这里需要根据实际数据逻辑确定)
            total_samples = len(test_data)
            seq_len = self.args['seq_len']
            full_length = 48  # Placeholder, ensure this matches your logic (e.g., total_data_points - seq_len + 1)

            # 遍历每个 batch
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args['data'] or 'Solar' in self.args['data']:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, :, :]).float()

                # 模型前向传播
                if self.args['use_amp']:
                    with torch.cuda.amp.autocast():
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = model_output[0] if self.args['output_attention'] else model_output
                else:
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = model_output[0] if self.args['output_attention'] else model_output

                f_dim = -1 if self.args['features'] == 'MS' else 0
                outputs = outputs[:, :, f_dim:]  # [batch_size, seq_len, 1] (归一化)
                batch_y = batch_y[:, :, f_dim:]  # [batch_size, seq_len, 1] (归一化)
                print('outputs_norm', outputs.shape, 'batch_y_norm', batch_y.shape)

                # --- 步骤 3.1: 计算基于 batch 的归一化损失 (模型尺度) ---
                rmse_batch_norm.update(rmse_loss(outputs, batch_y).item(), batch_x.size(0))
                mae_batch_norm.update(mae_loss(outputs, batch_y).item(), batch_x.size(0))

                # 反归一化数据 (Unnormalized)
                outputs_cpu = outputs.cpu().numpy().reshape(-1, 1)
                batch_y_cpu = batch_y.cpu().numpy().reshape(-1, 1)

                outputs_unnormalized = test_data.inverse_transform(outputs_cpu, is_target=True)
                batch_y_unnormalized = test_data.inverse_transform(batch_y_cpu, is_target=True)

                outputs_unnormalized_tensor = torch.from_numpy(outputs_unnormalized).float().reshape(outputs.shape)
                batch_y_unnormalized_tensor = torch.from_numpy(batch_y_unnormalized).float().reshape(batch_y.shape)

                print('batch_y_unnormalized', batch_y_unnormalized_tensor.shape, 'outputs_unnormalized',
                      outputs_unnormalized_tensor.shape)

                # --- ADDED: Print a sample of the unnormalized data for the first batch ---
                if i == 0:
                    print("\n--- 原始尺度数据样本 (仅限第一个 Batch) ---")
                    # 打印前2个样本点的预测和真实值
                    print("反归一化真实值 (batch_y_unnormalized_tensor[:2]):")
                    print(batch_y_unnormalized_tensor[:2].cpu().numpy())
                    print("反归一化预测值 (outputs_unnormalized_tensor[:2]):")
                    print(outputs_unnormalized_tensor[:2].cpu().numpy())
                    print("--------------------------------------------------\n")
                # --- END ADDITION ---

                # --- 步骤 3.2: 计算基于 batch 的反归一化损失 (原始尺度) ---
                rmse_batch_unnorm.update(rmse_loss(outputs_unnormalized_tensor, batch_y_unnormalized_tensor).item(),
                                         batch_x.size(0))
                mae_batch_unnorm.update(mae_loss(outputs_unnormalized_tensor, batch_y_unnormalized_tensor).item(),
                                        batch_x.size(0))

                # 保存归一化的数据用于去重叠 (Full Sequence Loss)
                all_outputs_norm.append(outputs_cpu)
                all_batch_y_norm.append(batch_y_cpu)

            # ----------------------------------------------------------------------
            #                       去重叠处理 (De-overlapping)
            # ----------------------------------------------------------------------

            # 初始化完整序列 (归一化数据)
            outputs_full = np.zeros((full_length, 1))
            batch_y_full = np.zeros((full_length, 1))
            counts = np.zeros(full_length)

            # 将所有 batch 的归一化结果填入完整序列
            sample_idx = 0
            for batch_outputs_norm, batch_y_norm in zip(all_outputs_norm, all_batch_y_norm):
                # 注意：这里可能需要根据实际的滑动窗口逻辑调整 batch_size 和 sample_idx 的计算
                # 这里的 batch_size = batch_outputs_norm.shape[0] // seq_len 假设样本是连续且不重叠的
                batch_size = batch_outputs_norm.shape[0] // seq_len
                for i in range(batch_size):
                    start_idx = sample_idx + i
                    end_idx = start_idx + seq_len
                    if end_idx <= full_length:  # 确保不越界
                        outputs_full[start_idx:end_idx] += batch_outputs_norm[i * seq_len:(i + 1) * seq_len]
                        batch_y_full[start_idx:end_idx] += batch_y_norm[i * seq_len:(i + 1) * seq_len]
                        counts[start_idx:end_idx] += 1
                sample_idx += batch_size

            # 对重叠部分取平均值 (得到去重叠后的归一化数据)
            outputs_full[counts > 0] = outputs_full[counts > 0] / counts[counts > 0, np.newaxis]
            batch_y_full[counts > 0] = batch_y_full[counts > 0] / counts[counts > 0, np.newaxis]

            # --- 步骤 4.1: 计算去重叠后的归一化损失 (模型尺度) ---
            outputs_full_norm_tensor = torch.from_numpy(outputs_full).float()
            batch_y_full_norm_tensor = torch.from_numpy(batch_y_full).float()
            rmse_full_norm.update(rmse_loss(outputs_full_norm_tensor, batch_y_full_norm_tensor).item(), full_length)
            mae_full_norm.update(mae_loss(outputs_full_norm_tensor, batch_y_full_norm_tensor).item(), full_length)

            # 反归一化去重叠后的数据 (Unnormalized, 原始尺度)
            outputs_full_unnormalized = test_data.inverse_transform(outputs_full, is_target=True)
            batch_y_full_unnormalized = test_data.inverse_transform(batch_y_full, is_target=True)

            # --- 步骤 4.2: 计算去重叠后的反归一化损失 (原始尺度) ---
            outputs_full_unnorm_tensor = torch.from_numpy(outputs_full_unnormalized).float()
            batch_y_full_unnorm_tensor = torch.from_numpy(batch_y_full_unnormalized).float()
            rmse_full_unnorm.update(rmse_loss(outputs_full_unnorm_tensor, batch_y_full_unnorm_tensor).item(),
                                    full_length)
            mae_full_unnorm.update(mae_loss(outputs_full_unnorm_tensor, batch_y_full_unnorm_tensor).item(), full_length)

            # ----------------------------------------------------------------------
            #                           结果保存与输出
            # ----------------------------------------------------------------------

            # 获取平均损失
            rmse_batch_norm_avg = rmse_batch_norm.avg
            mae_batch_norm_avg = mae_batch_norm.avg
            rmse_batch_unnorm_avg = rmse_batch_unnorm.avg
            mae_batch_unnorm_avg = mae_batch_unnorm.avg
            rmse_full_norm_avg = rmse_full_norm.avg
            mae_full_norm_avg = mae_full_norm.avg
            rmse_full_unnorm_avg = rmse_full_unnorm.avg
            mae_full_unnorm_avg = mae_full_unnorm.avg

            # 打印所有结果
            print("--------------------------------------------------")
            print("基于 Batch 的损失:")
            print(
                f"  [归一化 Loss] (Normalized, 模型尺度): RMSE: {rmse_batch_norm_avg:.4f}, MAE: {mae_batch_norm_avg:.4f}")
            print(
                f"  [反归一化 Loss] (Unnormalized, 原始尺度): RMSE: {rmse_batch_unnorm_avg:.4f}, MAE: {mae_batch_unnorm_avg:.4f}")
            print("\n去重叠后的完整序列损失:")
            print(
                f"  [归一化 Loss] (Normalized, 模型尺度): RMSE: {rmse_full_norm_avg:.4f}, MAE: {mae_full_norm_avg:.4f}")
            print(
                f"  [反归一化 Loss] (Unnormalized, 原始尺度): RMSE: {rmse_full_unnorm_avg:.4f}, MAE: {mae_full_unnorm_avg:.4f}")
            print("--------------------------------------------------")

            # --- Excel 保存部分 (保持与原始逻辑一致，仅使用反归一化后的去重叠数据) ---
            # 原始代码中基于 batch 的保存逻辑已删除以保持简洁，但保留了去重叠后的保存
            outputs_full_unnormalized = outputs_full_unnormalized.T
            batch_y_full_unnormalized = batch_y_full_unnormalized.T

            batch_y_columns = [f'batch_y_t{i}' for i in range(full_length)]
            outputs_columns = [f'outputs_t{i}' for i in range(full_length)]

            batch_y_df_full = pd.DataFrame(batch_y_full_unnormalized, columns=batch_y_columns)
            outputs_df_full = pd.DataFrame(outputs_full_unnormalized, columns=outputs_columns)

            results_df_full = pd.concat([batch_y_df_full, outputs_df_full], axis=0)

            save_path_full = r"D:\sea level variability\code_nepo\GRU - 12\output_test\Transformer.xlsx"  # 修改了文件名以示区分
            results_df_full.to_excel(save_path_full, index=False)

            # --- 生成 test_result 字符串并保存到 txt 文件 ---
            prefix = f"Epoch: {setting['epoch']}, lr={self.args['learning_rate']}" if isinstance(setting,
                                                                                                 dict) and 'epoch' in setting else f"lr={self.args['learning_rate']}"

            test_result_batch_norm = f"{prefix}, Batch Normalized (Model Scale): RMSE: {rmse_batch_norm_avg:.4f}, MAE: {mae_batch_norm_avg:.4f}"
            test_result_batch_unnorm = f"{prefix}, Batch Unnormalized (Original Scale): RMSE: {rmse_batch_unnorm_avg:.4f}, MAE: {mae_batch_unnorm_avg:.4f}"
            test_result_full_norm = f"{prefix}, Full Normalized (Model Scale): RMSE: {rmse_full_norm_avg:.4f}, MAE: {mae_full_norm_avg:.4f}"
            test_result_full_unnorm = f"{prefix}, Full Unnormalized (Original Scale): RMSE: {rmse_full_unnorm_avg:.4f}, MAE: {mae_full_unnorm_avg:.4f}"

            save_path_txt = os.path.join(r"D:\sea level variability\code_nepo\GRU - 12\SOFTS-main\test_result",
                                         'test_results.txt')
            with open(save_path_txt, 'a') as f:
                f.write(test_result_batch_norm + '\n')
                f.write(test_result_batch_unnorm + '\n')
                f.write(test_result_full_norm + '\n')
                f.write(test_result_full_unnorm + '\n')
                f.write('\n')

            return {
                'rmse_batch_norm_avg': rmse_batch_norm_avg,
                'mae_batch_norm_avg': mae_batch_norm_avg,
                'rmse_batch_unnorm_avg': rmse_batch_unnorm_avg,
                'mae_batch_unnorm_avg': mae_batch_unnorm_avg,
                'rmse_full_norm_avg': rmse_full_norm_avg,
                'mae_full_norm_avg': mae_full_norm_avg,
                'rmse_full_unnorm_avg': rmse_full_unnorm_avg,
                'mae_full_unnorm_avg': mae_full_unnorm_avg
            }
 