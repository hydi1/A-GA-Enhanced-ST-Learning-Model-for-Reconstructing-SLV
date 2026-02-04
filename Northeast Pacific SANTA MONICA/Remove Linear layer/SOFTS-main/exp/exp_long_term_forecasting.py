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
        #self.model_dict[self.args.model] 是一个模型字典，键为模型名称，值为对应的模型类。
        #.Model(self.args) 表示实例化该模型类并传入实验参数 args
        #.float() 将模型参数类型设置为 32 位浮点数，确保模型参数与输入数据的精度一致
        # 支持 model_dict 中存放模块（包含 Model 类）或直接存放 Model 类两种情况
        model_entry = self.model_dict[self.args['model']]
        if hasattr(model_entry, 'Model'):
            model = model_entry.Model(self.args).float()
        elif callable(model_entry):
            model = model_entry(self.args).float()
        else:
            raise TypeError(f"Unsupported model entry for {self.args['model']}: {type(model_entry)}")
        # print("self.model_dict[self.args['model']]", self.model_dict[self.args['model']])
        # print("模型的名称是",model)
        #如果 args.use_multi_gpu 和 args.use_gpu 都为 True，则使用 nn.DataParallel 将模型分布到多个 GPU 上运行。
        #device_ids 指定要使用的 GPU 设备列表
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

    def get_r2_eff(self, preds, targets):
        preds = np.array(preds).flatten()
        targets = np.array(targets).flatten()

        y_mean = np.mean(targets)
        ss_res = np.sum((preds - targets) ** 2)
        ss_tot = np.sum((targets - y_mean) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1.0 - ss_res / ss_tot
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

                # decoder input
                #创建一个与batch_y形状相同的零张量，用于解码器的输入 pred_len 96
                # dec_inp = torch.zeros_like(batch_y[:, -self.args['pred_len']:, :]).float()
                # #将batch_y的前label_len=48个时间步长的数据与dec_inp拼接，形成解码器的输入
                # dec_inp = torch.cat([batch_y[:, :self.args['label_len'], :], dec_inp], dim=1).float().to(self.device)

                # 解码器可以看到 batch_y 的全部前部分（label_len）
                dec_inp = batch_y.float().to(self.device)

                # print("dec_inp.shape", dec_inp.shape)#dec_inp.shape torch.Size([32, 96, 1])

                # encoder - decoder 编码器-解码器结构的前向计算
                #use_amp：是否使用自动混合精度训练
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
                #如果 self.args.features 为 'MS'，表示多变量输入，取最后一个特征维度
                f_dim = -1 if self.args['features'] == 'MS' else 0
                #从预测结果中取最后 pred_len 时间步的预测值。
                outputs = outputs[:, :, f_dim:]
                #从真实目标值中取最后 pred_len 时间步的目标值
                batch_y = batch_y[:, :, f_dim:].to(self.device)

                # print("torch.isnan(outputs).any(), torch.isnan(batch_y).any()",torch.isnan(outputs).any(), torch.isnan(batch_y).any())
                # print('torch.isinf(outputs).any(), torch.isinf(batch_y).any()',torch.isinf(outputs).any(), torch.isinf(batch_y).any())
                # print('outputs.min().item()',outputs.min().item(),'outputs.max().item()', outputs.max().item())

                loss = criterion(outputs, batch_y)
                # print("loss",loss)
                # assert not torch.isnan(loss), f"NaN in loss at step {i}"



                total_loss.update(loss.item(), batch_x.size(0))
        total_loss = total_loss.avg
        self.model.train()
        return total_loss



    def train(self, setting):
        #调用 self._get_data 方法，将输入数据（训练、验证、测试）封装成 Dataset 和 DataLoader
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        #保存模型权重文件的路径
        # print(f"Setting: {setting}")
        # print(f"self.args['checkpoints']: {self.args['checkpoints']}, type: {type(self.args['checkpoints'])}")
        # print(f"Setting: {setting}, type: {type(setting)}")
        first_13_values = list(setting.values())[:13]  # 取前13个值
        result_string = '_'.join(map(str, first_13_values))
        path = os.path.join(self.args['checkpoints'], result_string)


        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        print("共{}个训练批次".format(train_steps))
        # print("train_data的长度", len(train_data))#451
        # print("vali_data的长度", len(vali_data))#79
        # print("test_data的长度", len(test_data))#252-seq_len+1=157
        #print("train_steps", train_steps)#1075
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
                # print("train_batch_x shape:", batch_x.shape)#[32,96,441]
                #print("batch_x.shape",batch_x.shape)#[32,96,7] 96-输入时间序列的长度 由 arg.seq_len 决定 7：每个时间步的特征数，由 args.enc_in
                batch_y = batch_y.float().to(self.device)
                # print("train_batch_y shape:", batch_y.shape)#[32,96,1]
                #print("batch_y.shape", batch_y.shape)#[32,144,7] 144 -目标时间序列的长度，由 args.pred_len+label_len 决定
                # print("batch_x_mark.shape", batch_x_mark.shape)#torch.Size([32, 96, 1])
                # print("batch_y_mark.shape", batch_y_mark.shape)#torch.Size([32, 96, 1])
                if 'PEMS' in self.args['data'] or 'Solar' in self.args['data']:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args['pred_len']:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args['label_len'], :], dec_inp], dim=1).float().to(self.device)

                # 如果任务中不需要 pred_len，不使用 label_len 部分，二是直接使用上一个时间步的真实值
                #解码器需要输入上一时间步的真实值（batch_y）来生成当前时间步的预测值
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
                    # 修改了这里
                    # print("刚输出的outputs.shape", outputs.shape)
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    # print("loss", loss)


                # if (i + 1) % 100 == 0:
                    #当当前迭代次数是 100 的倍数时
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
                # 在反向传播后，梯度裁剪前检查梯度
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name} max grad: {param.grad.abs().max()}")

                # 梯度裁剪
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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        with torch.no_grad():
            all_outputs = []  # 保存所有预测结果（未反归一化）
            all_batch_y = []  # 保存所有真实值（未反归一化）

            # 遍历每个 batch，收集预测结果和真实值
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

                if self.args['use_amp']:
                    with torch.cuda.amp.autocast():
                        if self.args['output_attention']:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args['output_attention']:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args['features'] == 'MS' else 0
                outputs = outputs[:, :, f_dim:]  # [batch_size, seq_len, 1]
                batch_y = batch_y[:, :, f_dim:]  # [batch_size, seq_len, 1]

                # 未反归一化的数据 - 保持3维格式用于去重叠
                outputs_cpu = outputs.cpu().numpy()  # [batch_size, seq_len, 1]
                batch_y_cpu = batch_y.cpu().numpy()  # [batch_size, seq_len, 1]

                # 保存数据用于去重叠
                all_outputs.append(outputs_cpu)
                all_batch_y.append(batch_y_cpu)

            # 计算测试集的完整长度
            total_samples = len(test_data)
            seq_len = self.args['seq_len']
            full_length = total_samples + seq_len - 1

            # 初始化完整序列
            outputs_full = np.zeros((full_length, 1))
            batch_y_full = np.zeros((full_length, 1))
            counts = np.zeros(full_length)  # 记录每个时间点的预测次数

            # 将所有 batch 的结果填入完整序列（去重叠拼接）
            sample_global_idx = 0
            for b_out, b_y in zip(all_outputs, all_batch_y):  # b_out, b_y 都是 [batch_size, seq_len, 1]
                batch_size = b_out.shape[0]
                seq_len_batch = b_out.shape[1]
                for i in range(batch_size):
                    start_idx = sample_global_idx
                    end_idx = start_idx + seq_len_batch

                    actual_end = min(end_idx, full_length)
                    length_to_add = actual_end - start_idx

                    if length_to_add > 0:
                        outputs_full[start_idx:actual_end] += b_out[i, :length_to_add]  # [seq_len, 1]
                        batch_y_full[start_idx:actual_end] += b_y[i, :length_to_add]   # [seq_len, 1]
                        counts[start_idx:actual_end] += 1

                    sample_global_idx += 1

            # 对重叠部分求平均
            counts[counts == 0] = 1  # 避免除零
            outputs_full /= counts[:, np.newaxis]
            batch_y_full /= counts[:, np.newaxis]

            # =========================================================
            # ① 归一化尺度（Normalized）去重叠全序列指标
            # =========================================================
            rmse_norm_full = np.sqrt(np.mean((outputs_full - batch_y_full) ** 2))
            mae_norm_full = np.mean(np.abs(outputs_full - batch_y_full))
            r2_eff_norm_full = self.get_r2_eff(outputs_full, batch_y_full)

            # =========================================================
            # ② 反归一化去重叠后的数据
            # =========================================================
            outputs_full_unnormalized = test_data.inverse_transform(outputs_full, is_target=True)
            batch_y_full_unnormalized = test_data.inverse_transform(batch_y_full, is_target=True)

            # 反归一化尺度的去重叠指标
            rmse_denorm_full = np.sqrt(np.mean((outputs_full_unnormalized - batch_y_full_unnormalized) ** 2))
            mae_denorm_full = np.mean(np.abs(outputs_full_unnormalized - batch_y_full_unnormalized))
            r2_eff_denorm_full = self.get_r2_eff(outputs_full_unnormalized, batch_y_full_unnormalized)

            # 打印结果
            print("\n" + "=" * 50)
            print(f"去重叠后的测试结果 (Full Length={full_length}):")
            print("=" * 50)
            print(f"[Normalized]    RMSE: {rmse_norm_full:.4f} | MAE: {mae_norm_full:.4f} | R2_eff: {r2_eff_norm_full:.4f}")
            print(f"[De-normalized] RMSE: {rmse_denorm_full:.4f} | MAE: {mae_denorm_full:.4f}")
            print("=" * 50 + "\n")

            # 生成 test_result 字符串
            if isinstance(setting, dict) and 'epoch' in setting:
                result_str = f"Epoch: {setting['epoch']},seed={self.args['model_id']}, lr={self.args['learning_rate']}\n"
            else:
                result_str = f"lr={self.args['learning_rate']}\n,seed={self.args['model_id']}"

            # 保存到 txt 文件
            save_path_txt = os.path.join(r"D:\project\组件消融\东北太平洋\GAconvgru-移除Linear\SOFTS-main\test_result", 'test_results.txt')
            with open(save_path_txt, 'a') as f:
                f.write(result_str)
                f.write(f"[Normalized]    RMSE: {rmse_norm_full:.4f}, MAE: {mae_norm_full:.4f}, R2_eff: {r2_eff_norm_full:.4f}\n")
                f.write(f"[De-normalized] RMSE: {rmse_denorm_full:.4f}, MAE: {mae_denorm_full:.4f}\n")
                f.write('\n')  # 添加空行分隔

        return {
            'rmse_norm_full': rmse_norm_full,
            'mae_norm_full': mae_norm_full,
            'r2_eff_norm_full': r2_eff_norm_full,
            'rmse_denorm_full': rmse_denorm_full,
            'mae_denorm_full': mae_denorm_full
        }
