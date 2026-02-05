import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import joblib
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')

class Dataset_Npy(Dataset):
    def __init__(self, root_path, data_path, target_path, flag='train', size=None,
                 features='S', target='OT', scale=False, timeenc=0, freq='h', seasonal_patterns=None):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.target_path = target_path
        self.__read_data__()
        self.scaler_x = None
        self.scaler_y =None
        self.__read_data__()

    def __read_data__(self):
        print("Dataset scale =", self.scale, )

        data_raw = np.load(os.path.join(self.root_path, self.data_path))
        self.data_x = data_raw.reshape(-1, 4*29*6*11)

        df_target = pd.read_excel(self.target_path)
        self.data_y = df_target.iloc[:,-1].values
        print("data_y.shape=", self.data_y.shape)

        self.time = df_target.iloc[:, 0].values

        num_train = int(len(self.data_x) * 0.6)
        print("num_train=", num_train)
        num_test = int(len(self.data_x) * 0.1)
        print("num_test=", num_test)
        num_valid = int(len(self.data_x) * 0.3)
        print("num_valid=", num_valid)

        border1s = [0, num_train - self.seq_len, len(self.data_x) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(self.data_x)]
        border1 = border1s[self.set_type]

        border2 = border2s[self.set_type]

        flag_x = self.data_x[border1:border2]

        flag_y = self.data_y[border1:border2]

        if self.scale:
            train_data_x = data_raw.reshape(-1,4*29*6*11)[0:num_train]
            self.scaler_x = StandardScaler()

            train_data_x_fit = self.scaler_x.fit_transform(train_data_x)
            joblib.dump(self.scaler_x, 'scaler_x_time.pkl')
            if self.set_type == 0:
                self.data_x = train_data_x_fit
            elif self.set_type == 1:
                self.data_x = self.scaler_x.transform(flag_x)
            else:
                self.data_x = self.scaler_x.transform(flag_x)
        else:
            if self.set_type == 0:
                self.data_x = data_raw.reshape(-1,4*29*6*11)[0:num_train]
            elif self.set_type == 1:
                self.data_x = flag_x
            else:
                self.data_x = flag_x

        if self.scale:
            train_data_y = self.data_y[0:num_train]

            self.scaler_y = StandardScaler()
            train_data_y_fit = self.scaler_y.fit_transform(train_data_y.reshape(-1, 1))
            joblib.dump(self.scaler_y, 'scaler_y_time.pkl')
            if self.set_type == 0:
                self.data_y = train_data_y_fit
            elif self.set_type == 1:
                self.data_y = self.scaler_y.transform(flag_y.reshape(-1, 1))
                print("flag_y shape before transform:", flag_y.shape)

            else:

                print("flag_y shape before transform:", flag_y.shape)
                self.data_y = self.scaler_y.transform(flag_y.reshape(-1, 1))
        else:
            if self.set_type == 0:
                self.data_y = df_target.values[0:num_train]
            elif self.set_type == 1:
                self.data_y = flag_y
            else:
                self.data_y = flag_y

        df_stamp = pd.to_datetime(self.time)

        dates = pd.DatetimeIndex(df_stamp)

        self.data_stamp = time_features(dates, freq='D')
        self.data_stamp = self.data_stamp.transpose(1, 0)

    def __getitem__(self, index):
        seq_x = self.data_x[index:index + self.seq_len]

        seq_y = self.data_y[index:index + self.seq_len]

        seq_x_mark = self.data_stamp[index:index + self.seq_len]

        seq_y_mark = self.data_stamp[index:index + self.seq_len]

        if len(seq_y.shape) == 1:
            seq_y = seq_y[:, np.newaxis]

        return (

            seq_x,
            seq_y,
            seq_x_mark,
            seq_y_mark,
        )

    def __len__(self):

        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data, is_target=True):
        if is_target and self.scaler_y is None:
            raise ValueError("Scaler for target data has not been initialized.")
        if not is_target and self.scaler_x is None:
            raise ValueError("Scaler for input data has not been initialized.")
        if is_target:
            return self.scaler_y.inverse_transform(data)
        else:
            return self.scaler_x.inverse_transform(data)
