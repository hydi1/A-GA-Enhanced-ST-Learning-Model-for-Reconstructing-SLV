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

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader

    def _select_optimizer(self):

        model_optim = optim.Adam(self.model.parameters(), lr=self.args['learning_rate'],eps=1e-4)

        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):

        total_loss = AverageMeter()

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float()

                if 'PEMS' in self.args['data'] or 'Solar' in self.args['data']:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = batch_y.float().to(self.device)
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
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.update(loss.item(), batch_x.size(0))
        total_loss = total_loss.avg
        self.model.train()
        return total_loss

    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        first_13_values = list(setting.values())[:13]
        result_string = '_'.join(map(str, first_13_values))
        path = os.path.join(self.args['checkpoints'], result_string)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        print("Total {} Training batches".format(train_steps))
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

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'

        self.model.load_state_dict(torch.load(best_model_path))
        if not self.args['save_model']:
            import shutil
            shutil.rmtree(path)
        return self.model

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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        rmse_loss = RMSELoss()
        mae_loss = nn.L1Loss()

        rmse_batch_norm = AverageMeter()
        mae_batch_norm = AverageMeter()

        rmse_batch_unnorm = AverageMeter()
        mae_batch_unnorm = AverageMeter()

        rmse_full_norm = AverageMeter()
        mae_full_norm = AverageMeter()

        rmse_full_unnorm = AverageMeter()
        mae_full_unnorm = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            all_outputs_norm = []
            all_batch_y_norm = []

            total_samples = len(test_data)
            seq_len = self.args['seq_len']
            full_length = 48

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
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = model_output[0] if self.args['output_attention'] else model_output
                else:
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = model_output[0] if self.args['output_attention'] else model_output

                f_dim = -1 if self.args['features'] == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                print('outputs_norm', outputs.shape, 'batch_y_norm', batch_y.shape)

                rmse_batch_norm.update(rmse_loss(outputs, batch_y).item(), batch_x.size(0))
                mae_batch_norm.update(mae_loss(outputs, batch_y).item(), batch_x.size(0))

                outputs_cpu = outputs.cpu().numpy().reshape(-1, 1)
                batch_y_cpu = batch_y.cpu().numpy().reshape(-1, 1)

                outputs_unnormalized = test_data.inverse_transform(outputs_cpu, is_target=True)
                batch_y_unnormalized = test_data.inverse_transform(batch_y_cpu, is_target=True)

                outputs_unnormalized_tensor = torch.from_numpy(outputs_unnormalized).float().reshape(outputs.shape)
                batch_y_unnormalized_tensor = torch.from_numpy(batch_y_unnormalized).float().reshape(batch_y.shape)

                print('batch_y_unnormalized', batch_y_unnormalized_tensor.shape, 'outputs_unnormalized',
                      outputs_unnormalized_tensor.shape)

                if i == 0:
                    print("\n--- Original scale data samples (first Batch only) ---")

                    print("DenormalizationTrue values (batch_y_unnormalized_tensor[:2]):")
                    print(batch_y_unnormalized_tensor[:2].cpu().numpy())
                    print("Denormalized prediction值 (outputs_unnormalized_tensor[:2]):")
                    print(outputs_unnormalized_tensor[:2].cpu().numpy())
                    print("--------------------------------------------------\n")

                rmse_batch_unnorm.update(rmse_loss(outputs_unnormalized_tensor, batch_y_unnormalized_tensor).item(),
                                         batch_x.size(0))
                mae_batch_unnorm.update(mae_loss(outputs_unnormalized_tensor, batch_y_unnormalized_tensor).item(),
                                        batch_x.size(0))

                all_outputs_norm.append(outputs_cpu)
                all_batch_y_norm.append(batch_y_cpu)

            outputs_full = np.zeros((full_length, 1))
            batch_y_full = np.zeros((full_length, 1))
            counts = np.zeros(full_length)

            sample_idx = 0
            for batch_outputs_norm, batch_y_norm in zip(all_outputs_norm, all_batch_y_norm):

                batch_size = batch_outputs_norm.shape[0] // seq_len
                for i in range(batch_size):
                    start_idx = sample_idx + i
                    end_idx = start_idx + seq_len
                    if end_idx <= full_length:
                        outputs_full[start_idx:end_idx] += batch_outputs_norm[i * seq_len:(i + 1) * seq_len]
                        batch_y_full[start_idx:end_idx] += batch_y_norm[i * seq_len:(i + 1) * seq_len]
                        counts[start_idx:end_idx] += 1
                sample_idx += batch_size

            outputs_full[counts > 0] = outputs_full[counts > 0] / counts[counts > 0, np.newaxis]
            batch_y_full[counts > 0] = batch_y_full[counts > 0] / counts[counts > 0, np.newaxis]

            outputs_full_norm_tensor = torch.from_numpy(outputs_full).float()
            batch_y_full_norm_tensor = torch.from_numpy(batch_y_full).float()
            rmse_full_norm.update(rmse_loss(outputs_full_norm_tensor, batch_y_full_norm_tensor).item(), full_length)
            mae_full_norm.update(mae_loss(outputs_full_norm_tensor, batch_y_full_norm_tensor).item(), full_length)

            outputs_full_unnormalized = test_data.inverse_transform(outputs_full, is_target=True)
            batch_y_full_unnormalized = test_data.inverse_transform(batch_y_full, is_target=True)

            outputs_full_unnorm_tensor = torch.from_numpy(outputs_full_unnormalized).float()
            batch_y_full_unnorm_tensor = torch.from_numpy(batch_y_full_unnormalized).float()
            rmse_full_unnorm.update(rmse_loss(outputs_full_unnorm_tensor, batch_y_full_unnorm_tensor).item(),
                                    full_length)
            mae_full_unnorm.update(mae_loss(outputs_full_unnorm_tensor, batch_y_full_unnorm_tensor).item(), full_length)

            rmse_batch_norm_avg = rmse_batch_norm.avg
            mae_batch_norm_avg = mae_batch_norm.avg
            rmse_batch_unnorm_avg = rmse_batch_unnorm.avg
            mae_batch_unnorm_avg = mae_batch_unnorm.avg
            rmse_full_norm_avg = rmse_full_norm.avg
            mae_full_norm_avg = mae_full_norm.avg
            rmse_full_unnorm_avg = rmse_full_unnorm.avg
            mae_full_unnorm_avg = mae_full_unnorm.avg

            print("--------------------------------------------------")
            print("基于 Batch 的损失:")
            print(
                f"  [Normalization Loss] (Normalized, 模型尺度): RMSE: {rmse_batch_norm_avg:.4f}, MAE: {mae_batch_norm_avg:.4f}")
            print(
                f"  [Denormalization Loss] (Unnormalized, 原始尺度): RMSE: {rmse_batch_unnorm_avg:.4f}, MAE: {mae_batch_unnorm_avg:.4f}")
            print("\nDe-overlapping后的完整序列损失:")
            print(
                f"  [Normalization Loss] (Normalized, 模型尺度): RMSE: {rmse_full_norm_avg:.4f}, MAE: {mae_full_norm_avg:.4f}")
            print(
                f"  [Denormalization Loss] (Unnormalized, 原始尺度): RMSE: {rmse_full_unnorm_avg:.4f}, MAE: {mae_full_unnorm_avg:.4f}")
            print("--------------------------------------------------")

            outputs_full_unnormalized = outputs_full_unnormalized.T
            batch_y_full_unnormalized = batch_y_full_unnormalized.T

            batch_y_columns = [f'batch_y_t{i}' for i in range(full_length)]
            outputs_columns = [f'outputs_t{i}' for i in range(full_length)]

            batch_y_df_full = pd.DataFrame(batch_y_full_unnormalized, columns=batch_y_columns)
            outputs_df_full = pd.DataFrame(outputs_full_unnormalized, columns=outputs_columns)

            results_df_full = pd.concat([batch_y_df_full, outputs_df_full], axis=0)

            save_path_full = r"D:\sea level variability\code_nepo\GRU - 12\output_test\Transformer.xlsx"
            results_df_full.to_excel(save_path_full, index=False)

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
