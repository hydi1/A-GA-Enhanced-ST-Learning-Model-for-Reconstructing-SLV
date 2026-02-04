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
def module_grad_norm(module: torch.nn.Module) -> float:
    # 计算一个模块所有参数梯度的整体L2范数
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += g.pow(2).sum().item()
    return total ** 0.5


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
        # model = self.model_dict[self.args.model].Model(self.args).float()
        model = self.model_dict[self.args['model']].Model(self.args).float()
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
    #用于验证模型性能的函数 它通过遍历验证数据集计算模型的平均损失（total_loss),用于衡量模型在验证集上的表现
    # 在 Exp_Long_Term_Forecast 类内部
    def get_r2_eff(self, preds, targets):
        preds = np.array(preds).flatten()
        targets = np.array(targets).flatten()

        y_mean = np.mean(targets)
        ss_res = np.sum((preds - targets) ** 2)
        ss_tot = np.sum((targets - y_mean) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1.0 - ss_res / ss_tot
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
        # 是否记录梯度范数（跨 epoch 累积），由 args['record_grad'] 控制
        record_grad = bool(self.args.get('record_grad', False))
        if record_grad:
            grad_gru_list = []
            grad_first_list = []
        else:
            grad_gru_list = None
            grad_first_list = None
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
                    if record_grad:
                        model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                        try:
                            grad_gru = module_grad_norm(model_ref.gru)
                        except Exception:
                            grad_gru = float('nan')
                        if hasattr(model_ref, 'qconvGRU'):
                            try:
                                grad_first = module_grad_norm(model_ref.qconvGRU)
                            except Exception:
                                grad_first = float('nan')
                        else:
                            try:
                                grad_first = module_grad_norm(model_ref.convGRU)
                            except Exception:
                                grad_first = float('nan')
                        grad_gru_list.append(grad_gru)
                        grad_first_list.append(grad_first)
                        print(f"Grad(gru)={grad_gru:.6e}, Grad(first)={grad_first:.6e}")

                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if record_grad:
                        model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                        try:
                            grad_gru = module_grad_norm(model_ref.gru)
                        except Exception:
                            grad_gru = float('nan')
                        if hasattr(model_ref, 'qconvGRU'):
                            try:
                                grad_first = module_grad_norm(model_ref.qconvGRU)
                            except Exception:
                                grad_first = float('nan')
                        else:
                            try:
                                grad_first = module_grad_norm(model_ref.convGRU)
                            except Exception:
                                grad_first = float('nan')
                        grad_gru_list.append(grad_gru)
                        grad_first_list.append(grad_first)
                        print(f"Grad(gru)={grad_gru:.6e}, Grad(first)={grad_first:.6e}")

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

        # 保存梯度统计到文件（跨所有 epoch），仅当启用时保存
        try:
            if record_grad and grad_gru_list is not None and len(grad_gru_list) > 0:
                grad_arr = np.vstack([np.array(grad_gru_list), np.array(grad_first_list)]).T
                grad_path = os.path.join(path, 'grad_stats.csv')
                np.savetxt(grad_path, grad_arr, header='grad_gru,grad_first', delimiter=',', comments='')
        except Exception:
            pass

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
        all_outputs = []
        all_batch_y = []

        with torch.no_grad():
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
                # 保持 [Batch, Seq, Dim]
                outputs = outputs[:, :, f_dim:].cpu().numpy()
                batch_y = batch_y[:, :, f_dim:].cpu().numpy()

                all_outputs.append(outputs)
                all_batch_y.append(batch_y)

        # --- 去重叠处理逻辑 (De-overlapping) ---
        total_samples = len(test_data)  # 37
        seq_len = self.args['seq_len']  # 12
        full_length = 48  # 按照你的要求固定为 48

        outputs_full = np.zeros((full_length, 1))
        batch_y_full = np.zeros((full_length, 1))
        counts = np.zeros(full_length)

        # 逐样本填充
        sample_global_idx = 0
        for b_out, b_y in zip(all_outputs, all_batch_y):
            batch_size = b_out.shape[0]
            for i in range(batch_size):
                start_idx = sample_global_idx
                end_idx = start_idx + seq_len

                # 边界保护
                actual_end = min(end_idx, full_length)
                length_to_add = actual_end - start_idx

                if length_to_add > 0:
                    outputs_full[start_idx:actual_end] += b_out[i, :length_to_add, :]
                    batch_y_full[start_idx:actual_end] += b_y[i, :length_to_add, :]
                    counts[start_idx:actual_end] += 1

                sample_global_idx += 1

        # 对重叠部分取平均
        counts[counts == 0] = 1
        outputs_full /= counts[:, np.newaxis]
        batch_y_full /= counts[:, np.newaxis]

        # ==================================================
        # ① 新增：归一化尺度 RMSE / MAE
        # ==================================================
        rmse_norm = np.sqrt(np.mean((outputs_full - batch_y_full) ** 2))
        mae_norm = np.mean(np.abs(outputs_full - batch_y_full))
        # ✅ 新增：归一化尺度 + 去重叠全序列 R2_eff
        r2_eff_norm_full = self.get_r2_eff(outputs_full, batch_y_full)

        # --- 反归一化 ---
        outputs_final = test_data.inverse_transform(outputs_full, is_target=True)
        targets_final = test_data.inverse_transform(batch_y_full, is_target=True)

        # --- 计算最终全局指标（反归一化尺度） ---
        rmse = np.sqrt(np.mean((outputs_final - targets_final) ** 2))
        mae = np.mean(np.abs(outputs_final - targets_final))

        # --- 保存去重叠后的数据到 Excel ---
        out_save = outputs_final.T
        tar_save = targets_final.T
        col_names = [f't{i}' for i in range(full_length)]
        df_tar = pd.DataFrame(tar_save, columns=col_names, index=['GroundTruth'])
        df_out = pd.DataFrame(out_save, columns=col_names, index=['Prediction'])
        results_df_full = pd.concat([df_tar, df_out], axis=0)

        save_path_full = r"D:\sea level variability\code_nepo\组件消融\东北太平洋组件消融+其他模型\GAconvgru替换为convgru\output_test\output_y12_deoverlapped.xlsx"
        results_df_full.to_excel(save_path_full)

        # --- 打印与记录 ---
        print("\n" + "=" * 45)
        print(f"Final Global Metrics (De-overlapped, Full={full_length})")
        print(f"[Normalized]    RMSE: {rmse_norm:.4f} | MAE: {mae_norm:.4f} | R2_eff: {r2_eff_norm_full:.4f}")
        print(f"[De-normalized] RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        print("=" * 45)

        # 保存到 txt 文件（新增：归一化指标 + 学习率）
        save_path_txt = os.path.join(
            r"D:\sea level variability\code_nepo\组件消融\东北太平洋组件消融+其他模型\GAconvgru替换为convgru\SOFTS-main\test_result",
            'test_results.txt'
        )
        with open(save_path_txt, 'a') as f:
            f.write(f"seed={self.args['model_id']} | lr={self.args['learning_rate']}\n")
            f.write(f"[Normalized]    RMSE: {rmse_norm:.4f}, MAE: {mae_norm:.4f}, R2_eff: {r2_eff_norm_full:.4f}\n")
            f.write(f"[De-normalized] RMSE: {rmse:.4f}, MAE: {mae:.4f}\n\n")

        return {
            'rmse_norm': rmse_norm,
            'mae_norm': mae_norm,
            'r2_eff_norm_full': r2_eff_norm_full,
            'rmse': rmse,
            'mae': mae
        }


