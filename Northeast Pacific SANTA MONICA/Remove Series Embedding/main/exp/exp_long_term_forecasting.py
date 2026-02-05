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
                outputs = outputs[:, :, f_dim:].cpu().numpy()
                batch_y = batch_y[:, :, f_dim:].cpu().numpy()

                all_outputs.append(outputs)
                all_batch_y.append(batch_y)

        total_samples = len(test_data)
        seq_len = self.args['seq_len']
        full_length = total_samples + seq_len - 1

        outputs_full = np.zeros((full_length, 1))
        batch_y_full = np.zeros((full_length, 1))
        counts = np.zeros(full_length)

        sample_global_idx = 0
        for b_out, b_y in zip(all_outputs, all_batch_y):
            batch_size = b_out.shape[0]
            for i in range(batch_size):
                start_idx = sample_global_idx
                end_idx = start_idx + seq_len

                actual_end = min(end_idx, full_length)
                length_to_add = actual_end - start_idx

                if length_to_add > 0:
                    outputs_full[start_idx:actual_end] += b_out[i, :length_to_add, :]
                    batch_y_full[start_idx:actual_end] += b_y[i, :length_to_add, :]
                    counts[start_idx:actual_end] += 1

                sample_global_idx += 1

        counts[counts == 0] = 1
        outputs_full /= counts[:, np.newaxis]
        batch_y_full /= counts[:, np.newaxis]

        rmse_norm = np.sqrt(np.mean((outputs_full - batch_y_full) ** 2))
        mae_norm = np.mean(np.abs(outputs_full - batch_y_full))

        r2_eff_norm_full = self.get_r2_eff(outputs_full, batch_y_full)

        outputs_final = test_data.inverse_transform(outputs_full, is_target=True)
        targets_final = test_data.inverse_transform(batch_y_full, is_target=True)

        if outputs_final.ndim == 1:
            outputs_final = outputs_final.reshape(-1, 1)
        if targets_final.ndim == 1:
            targets_final = targets_final.reshape(-1, 1)

        T, D = outputs_final.shape
        data_dict = {"time_idx": np.arange(T)}

        if D == 1:
            data_dict["pred"] = outputs_final[:, 0]
            data_dict["true"] = targets_final[:, 0]
        else:
            for d in range(D):
                data_dict[f"pred_{d}"] = outputs_final[:, d]
            for d in range(D):
                data_dict[f"true_{d}"] = targets_final[:, d]

        df = pd.DataFrame(data_dict)

        save_dir = r"D:\project\组件消融\东北太平洋\GAconvgru-移除DataEmbedding\output_test"
        os.makedirs(save_dir, exist_ok=True)
        save_path_excel = os.path.join(save_dir, f"{setting}_deoverlap_denorm.xlsx")
        df.to_excel(save_path_excel, index=False)

        rmse = np.sqrt(np.mean((outputs_final - targets_final) ** 2))
        mae = np.mean(np.abs(outputs_final - targets_final))

        print("\n" + "=" * 60)
        print(f"Final Global Metrics (De-overlapped, Full Length={full_length}):")
        print(f"[Normalized]    RMSE: {rmse_norm:.4f} | MAE: {mae_norm:.4f} | R2_eff(full): {r2_eff_norm_full:.4f}")
        print(f"[De-normalized] RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        print("=" * 60)

        save_path_txt = os.path.join(
            r"D:\project\组件消融\东北太平洋\GAconvgru-移除DataEmbedding\SOFTS-main\test_result",
            'test_results.txt'
        )
        with open(save_path_txt, 'a') as f:
            f.write(f"seed={self.args['model_id'] if 'model_id' in self.args else 'default'} | lr={self.args['learning_rate']}\n")
            f.write(f"[Normalized]    RMSE: {rmse_norm:.4f}, MAE: {mae_norm:.4f}, R2_eff(full): {r2_eff_norm_full:.4f}\n")
            f.write(f"[De-normalized] RMSE: {rmse:.4f}, MAE: {mae:.4f}\n\n")

        return {
            'rmse_norm': rmse_norm,
            'mae_norm': mae_norm,
            'r2_eff_norm_full': r2_eff_norm_full,
            'rmse': rmse,
            'mae': mae
        }
