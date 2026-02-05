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

    def get_r2_eff(self, preds, targets):
        """
        Calculate effective explanation rate (R²) - Applicable to De-overlapping Denormalization Test set data

        有效解释率的定义：
        R² = 1 - (SS_res / SS_tot)
        其中：
        - SS_res: Residual Sum of Squares = Σ(y_pred - y_true)²
        - SS_tot: Total Sum of Squares = Σ(y_true - y_mean)²
        - R² ∈ (-∞, 1]：Values closer to 1 indicate better model fit; negative values indicate model performance worse than mean model

        Args:
            preds (array-like): Predicted values，形状为 (N,) 或 (N, 1)
            targets (array-like): True target values, shape (N,) or (N, 1)

        Returns:
            float: 有效解释率 R² 值

        Example:
            >>> exp = Exp_Long_Term_Forecast(args)
            >>> preds_full = outputs_full_unnormalized  # De-overlappingDenormalized predicted values
            >>> targets_full = batch_y_full_unnormalized  # De-overlappingDenormalized true values
            >>> r2 = exp.get_r2_eff(preds_full, targets_full)
            >>> print(f"R²: {r2:.4f}")
        """

        preds = np.asarray(preds).flatten()
        targets = np.asarray(targets).flatten()

        y_mean = np.mean(targets)

        ss_res = np.sum((preds - targets) ** 2)

        ss_tot = np.sum((targets - y_mean) ** 2)

        if ss_tot == 0:

            if ss_res == 0:
                return 1.0

            return 0.0

        r2_eff = 1.0 - (ss_res / ss_tot)

        return r2_eff

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
                print("outputs.shape", outputs.shape)

                batch_y = batch_y[:, :, f_dim:].to(self.device)
                print("batch_y.shape", batch_y.shape)

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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        rmse_loss = RMSELoss()
        mae_loss = nn.L1Loss()

        rmse_full_norm = AverageMeter()
        mae_full_norm = AverageMeter()
        r2_full_norm = AverageMeter()

        rmse_full_denorm = AverageMeter()
        mae_full_denorm = AverageMeter()

        rmse_batch_norm = AverageMeter()
        mae_batch_norm = AverageMeter()
        r2_batch_norm = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            all_outputs = []
            all_batch_y = []

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
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                print('outputs', outputs.shape, 'batch_y', batch_y.shape)

                outputs_flat = outputs.reshape(-1, 1)
                batch_y_flat = batch_y.reshape(-1, 1)
                n_batch = outputs_flat.numel()

                rmse_batch_norm.update(rmse_loss(outputs_flat, batch_y_flat).item(), n_batch)
                mae_batch_norm.update(mae_loss(outputs_flat, batch_y_flat).item(), n_batch)

                outputs_np = outputs_flat.detach().cpu().numpy()
                batch_y_np = batch_y_flat.detach().cpu().numpy()
                r2_eff_batch = self.get_r2_eff(outputs_np, batch_y_np)
                r2_batch_norm.update(r2_eff_batch, n_batch)

                all_outputs.append(outputs_np)
                all_batch_y.append(batch_y_np)

            total_samples = len(test_data)
            print('total_samples', total_samples)
            seq_len = self.args['seq_len']
            full_length = total_samples + seq_len - 1
            print('full_length', full_length)

            outputs_full = np.zeros((full_length, 1))
            batch_y_full = np.zeros((full_length, 1))
            counts = np.zeros(full_length)

            sample_idx = 0
            for batch_outputs, batch_y_arr in zip(all_outputs, all_batch_y):
                batch_size = batch_outputs.shape[0] // seq_len
                for j in range(batch_size):
                    start_idx = sample_idx + j
                    end_idx = start_idx + seq_len

                    outputs_full[start_idx:end_idx] += batch_outputs[j * seq_len:(j + 1) * seq_len]
                    batch_y_full[start_idx:end_idx] += batch_y_arr[j * seq_len:(j + 1) * seq_len]
                    counts[start_idx:end_idx] += 1

                sample_idx += batch_size

            counts[counts == 0] = 1
            outputs_full = outputs_full / counts[:, np.newaxis]
            batch_y_full = batch_y_full / counts[:, np.newaxis]

            outputs_full_norm_tensor = torch.from_numpy(outputs_full).float()
            batch_y_full_norm_tensor = torch.from_numpy(batch_y_full).float()

            rmse_full_norm.update(
                rmse_loss(outputs_full_norm_tensor, batch_y_full_norm_tensor).item(),
                full_length
            )
            mae_full_norm.update(
                mae_loss(outputs_full_norm_tensor, batch_y_full_norm_tensor).item(),
                full_length
            )

            r2_eff_full_norm = self.get_r2_eff(outputs_full, batch_y_full)
            r2_full_norm.update(r2_eff_full_norm, full_length)

            outputs_full_unnormalized = test_data.inverse_transform(outputs_full, is_target=True)
            batch_y_full_unnormalized = test_data.inverse_transform(batch_y_full, is_target=True)

            outputs_full_denorm_tensor = torch.from_numpy(outputs_full_unnormalized).float()
            batch_y_full_denorm_tensor = torch.from_numpy(batch_y_full_unnormalized).float()

            rmse_full_denorm.update(
                rmse_loss(outputs_full_denorm_tensor, batch_y_full_denorm_tensor).item(),
                full_length
            )
            mae_full_denorm.update(
                mae_loss(outputs_full_denorm_tensor, batch_y_full_denorm_tensor).item(),
                full_length
            )

            all_batch_y_np = np.concatenate(all_batch_y, axis=0)
            all_outputs_np = np.concatenate(all_outputs, axis=0)

            all_batch_y_reshaped = all_batch_y_np.reshape(total_samples, -1)
            all_outputs_reshaped = all_outputs_np.reshape(total_samples, -1)

            batch_y_df = pd.DataFrame(all_batch_y_reshaped, columns=[f'batch_y_step_{i + 1}' for i in range(seq_len)])
            outputs_df = pd.DataFrame(all_outputs_reshaped, columns=[f'outputs_step_{i + 1}' for i in range(seq_len)])
            results_df = pd.concat([batch_y_df, outputs_df], axis=1)

            save_path_batch = r"D:\sea level variability\code_eio\消融实验\GAconvGRU替换为convgru\output_test\output_y12.xlsx"
            results_df.to_excel(save_path_batch, index=False)

            outputs_full_unnormalized_T = outputs_full_unnormalized.T
            batch_y_full_unnormalized_T = batch_y_full_unnormalized.T

            batch_y_columns = [f'batch_y_t{i}' for i in range(full_length)]
            outputs_columns = [f'outputs_t{i}' for i in range(full_length)]

            batch_y_df_full = pd.DataFrame(batch_y_full_unnormalized_T, columns=batch_y_columns)
            outputs_df_full = pd.DataFrame(outputs_full_unnormalized_T, columns=outputs_columns)

            results_df_full = pd.concat([batch_y_df_full, outputs_df_full], axis=0)
            save_path_full = r"D:\sea level variability\code_eio\消融实验\GAconvGRU替换为convgru\output_test\output_y12_deoverlapped.xlsx"
            results_df_full.to_excel(save_path_full, index=False)

            rmse_batch_norm_avg = rmse_batch_norm.avg
            mae_batch_norm_avg = mae_batch_norm.avg
            r2_batch_norm_avg = r2_batch_norm.avg

            rmse_full_norm_avg = rmse_full_norm.avg
            mae_full_norm_avg = mae_full_norm.avg
            r2_full_norm_avg = r2_full_norm.avg

            rmse_full_denorm_avg = rmse_full_denorm.avg
            mae_full_denorm_avg = mae_full_denorm.avg

            if isinstance(setting, dict) and 'epoch' in setting:
                test_result_batch_norm = (
                    f"Epoch: {setting['epoch']}, lr={self.args['learning_rate']}, "
                    f"Batch Normalized: rmse: {rmse_batch_norm_avg:.4f}, "
                    f"mae: {mae_batch_norm_avg:.4f}, r2_eff: {r2_batch_norm_avg:.4f}"
                )
                test_result_norm = (
                    f"Epoch: {setting['epoch']}, lr={self.args['learning_rate']}, "
                    f"Deoverlap Normalized: rmse: {rmse_full_norm_avg:.4f}, "
                    f"mae: {mae_full_norm_avg:.4f}, r2_eff: {r2_full_norm_avg:.4f}"
                )
                test_result_denorm = (
                    f"Epoch: {setting['epoch']}, lr={self.args['learning_rate']}, "
                    f"Deoverlap Denormalized: rmse: {rmse_full_denorm_avg:.4f}, "
                    f"mae: {mae_full_denorm_avg:.4f}"
                )
            else:
                test_result_batch_norm = (
                    f"lr={self.args['learning_rate']}, "
                    f"Batch Normalized: rmse: {rmse_batch_norm_avg:.4f}, "
                    f"mae: {mae_batch_norm_avg:.4f}, r2_eff: {r2_batch_norm_avg:.4f}"
                )
                test_result_norm = (
                    f"lr={self.args['learning_rate']}, "
                    f"Deoverlap Normalized: rmse: {rmse_full_norm_avg:.4f}, "
                    f"mae: {mae_full_norm_avg:.4f}, r2_eff: {r2_full_norm_avg:.4f}"
                )
                test_result_denorm = (
                    f"lr={self.args['learning_rate']}, "
                    f"Deoverlap Denormalized: rmse: {rmse_full_denorm_avg:.4f}, "
                    f"mae: {mae_full_denorm_avg:.4f}"
                )

            save_path_txt = os.path.join(
                r"D:\sea level variability\code_eio\消融实验\GAconvGRU替换为convgru\SOFTS-main\test_result",
                'test_results.txt'
            )
            with open(save_path_txt, 'a') as f:
                f.write(f"seed={self.args['model_id']} | lr={self.args['learning_rate']}\n")
                f.write(test_result_batch_norm + '\n')
                f.write(test_result_norm + '\n')
                f.write(test_result_denorm + '\n')
                f.write('\n')

        return {

            'rmse_batch_norm_avg': rmse_batch_norm_avg,
            'mae_batch_norm_avg': mae_batch_norm_avg,
            'r2_batch_norm_avg': r2_batch_norm_avg,

            'rmse_full_norm_avg': rmse_full_norm_avg,
            'mae_full_norm_avg': mae_full_norm_avg,
            'r2_full_norm_avg': r2_full_norm_avg,
            'rmse_full_denorm_avg': rmse_full_denorm_avg,
            'mae_full_denorm_avg': mae_full_denorm_avg
        }
