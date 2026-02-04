import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
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
        self.__read_data__()

    def __read_data__(self):
        #定义归一化
        self.scaler = StandardScaler()
        #从指定路径加载数据
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        #M 或MS使用多变量特征
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            #S 仅适用单变量特征
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len#这里是sex_x的时间步范围
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len#这里是为什么 sex_y的时间步范围是[r_begin:r_red]

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        #(96,7)(144,7)(96,4)(144,4)
        #print("seq_x.shape,seq_y.shape,seq_x_mark.shape,seq_y_mark.shape",seq_x.shape,seq_y.shape,seq_x_mark.shape,seq_y_mark.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
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
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]#train 时 这里是0
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):# DataLoader 中的 __getitem__ 方法 会自动传递变化deindex
        s_begin = index#0
        s_end = s_begin + self.seq_len#96
        r_begin = s_end - self.label_len#96-48=48
        r_end = r_begin + self.label_len + self.pred_len#48+48+96=192

        seq_x = self.data_x[s_begin:s_end]#【0：96】
        seq_y = self.data_y[r_begin:r_end]#【48：192】
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # print("seq_x.shape,seq_y.shape,seq_x_mark.shape,seq_y_mark.shape", seq_x_mark.shape,
              # seq_y_mark.shape)(96, 7) (96, 7) (96, 4) (96, 4)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
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
        self.__read_data__()

    # def __read_data__(self):
    #     self.scaler = StandardScaler()
    #     df_raw = pd.read_csv(os.path.join(self.root_path,
    #                                       self.data_path))
    def __read_data__(self):
        self.scaler = StandardScaler()

        # Check if the data file is a .npy file
        if self.data_path.endswith('.npy'):
            # Load the .npy file using numpy
            df_raw = np.load(os.path.join(self.root_path, self.data_path))
        else:
            # If it's a CSV file, read it using pandas
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))


        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index#0
        s_end = s_begin + self.seq_len#96
        r_begin = s_end - self.label_len#96-48
        r_end = r_begin + self.label_len + self.pred_len#48+48+96

        seq_x = self.data_x[s_begin:s_end]#【0:96】
        seq_y = self.data_y[r_begin:r_end]#【48:192】
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Random(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
            self.n_channel = 512
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            self.n_channel = size[3]
        # init
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
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.DataFrame(np.random.rand(10000 + self.seq_len + self.pred_len, self.n_channel + 1),
                              columns=['f' + str(i) for i in range(self.n_channel + 1)])

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 4))
        seq_y_mark = torch.zeros((seq_y.shape[0], 4))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
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
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
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
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ssta.npy',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='M', cols=None, **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        #使用 StandardScaler 对数据进行归一化
        self.scaler = StandardScaler()
        #从指定路径加载数据
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))
        data_raw = np.load(os.path.join(self.root_path, self.data_path))
        df_raw  = data_raw.reshape(-1, 21 * 21)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        #提取所需要的列：如果指定了self.cols，从中移除目标列 target,只保留其余的特征列
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:#如果没有指定cols默认选择所有非目标列和非时间列作为特征
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        #将时间列（date）、特征列和目标列按照顺序排列，生成新的 DataFrame
        df_raw = df_raw[['date'] + cols + [self.target]]
        #窗口起始位置，是总数据长度减去序列长度 seq_len
        border1 = len(df_raw) - self.seq_len
        #窗口终止位置，等于数据的总长度
        border2 = len(df_raw)
        #如果 features 为 'M' 或 'MS'，使用所有特征列（多变量特征），如果 features 为 'S'，只使用目标列 target（单变量特征）
        # 将数据提取为一个新的 DataFrame，用于建模
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        #如果启用 scale，对数据进行标准化
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:#data 最终是一个二维数组，每行表示一个时间点的特征值
            data = df_data.values
        #提取窗口范围内的时间戳（tmp_stamp），并将字符串转换为 datetime 对象
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        #生成未来预测的时间戳 pred_dates，开始时间为窗口最后一个时间点，总长度为 pred_len + 1（包含第一个时间点）
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)
        #合并历史时间戳（tmp_stamp）和预测时间戳（pred_dates
        df_stamp = pd.DataFrame(columns=['date'])
        #预测时间戳从 pred_dates[1:] 开始
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        #时间特征生成
        #如果 timeenc == 0，手动生成月份、日期、星期几、小时等基本时间特征
        #如果 timeenc == 1，调用 time_features 函数，生成高级时间特征（如周期性特征）
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        #分离数据 
        #将数据划分为 输入数据和目标数据
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
            #时间特征保存在data_stamp中
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        #输入序列的结束位置
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    #总样本数=总数据点数-输入序列长度+1
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Npy(Dataset):
    def __init__(self, root_path, data_path, target_path, flag='train', size=None,
                 features='S', target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        # init
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
        self.scaler_x = None # 用于输入数据 x 的 scaler
        self.scaler_y =None # 用于目标数据 y 的 scaler

    def __read_data__(self):

        data_raw = np.load(os.path.join(self.root_path, self.data_path))
        self.data_x = data_raw.reshape(-1, 21 * 21) # 输入特征

        num_train = int(len(self.data_x) * 0.7)
        # print("num_train=", num_train)#546
        num_test = int(len(self.data_x) * 0.2)
        # print("num_test=", num_test)#156
        num_valid = int(len(self.data_x) * 0.1)
        # print("num_valid=", num_valid)#78
        #数据集是按时间顺序切分的
        border1s = [0, num_train - self.seq_len, len(self.data_x) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(self.data_x)]
        border1 = border1s[self.set_type]
        # print("border1=", border1s[0], border1s[1], border1s[2])#border1= 0 450 529
        border2 = border2s[self.set_type]
        # print("border2=", border2s[0], border2s[1], border2s[2])#border2= 546 624 781

        # 对x归一化
        # df_data = self.data_x
        if self.scale:
            self.scaler_x = MinMaxScaler()
            train_data_x = self.data_x[border1s[0]:border2s[0]]
            self.scaler_x.fit(train_data_x)
            data_x = self.scaler_x.transform(self.data_x)
        else:
            data_x = self.data_x

        # 从 Excel 文件中读取目标数据
        df_target = pd.read_excel(os.path.join(self.root_path, self.target_path), header=None)
        self.data_y = df_target.values  # 目标数据

        # import joblib
        #
        # scaler_path = r"D:\project\2SOFTS-main\SOFTS-main\scaler_y.pkl"  # 指定保存的路径
        #
        # if self.set_type == 0:  # 训练集阶段
        #     if self.scale:
        #         # 归一化
        #         self.scaler_y = MinMaxScaler()
        #         train_data_y = self.data_y[border1s[0]:border2s[0]]
        #         self.scaler_y.fit(train_data_y)  # 拟合训练数据
        #         data_y = self.scaler_y.transform(self.data_y)  # 转换数据
        #
        #         # 保存 scaler_y 到文件
        #         joblib.dump(self.scaler_y, scaler_path)  # 保存 scaler_y
        #
        #     else:
        #         data_y = self.data_y
        #
        # elif self.set_type == 1 or self.set_type == 2:  # 验证集或测试集阶段
        #     if self.scale:
        #         # 加载之前保存的 scaler_y
        #         self.scaler_y = joblib.load(scaler_path)
        #
        #         # 使用已加载的 scaler_y 对数据进行转换
        #         data_y = self.scaler_y.transform(self.data_y)
        #     else:
        #         data_y = self.data_y

        self.data_x = data_x[border1:border2]
        # print("data_x.shape=", self.data_x.shape)#(546,441)
        self.data_y = data_y[border1:border2]
        # print("data_y.shape=", self.data_y.shape)#(546,1)


    def __getitem__(self, index):
        seq_x = self.data_x[index:index + self.seq_len]  # 输入序列
        # print("seq_x.shape=", seq_x.shape)#(seq_len,441)
        seq_y = self.data_y[index:index + self.seq_len]  # 目标序列，长度与 seq_x 相同
        # print("seq_y.shape=", seq_y.shape)#(seq_len,1)
        # 如果有时间标记需求，用实际数据替代全零张量
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))  # 根据实际情况修改
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))  # 根据实际情况修改

        # 确保 seq_y 为二维 [seq_len, 1]
        if len(seq_y.shape) == 1:
            seq_y = seq_y[:, np.newaxis]

        return (
            torch.from_numpy(seq_x).float(),
            torch.from_numpy(seq_y).float(),
            seq_x_mark,
            seq_y_mark,
        )

    def __len__(self):
        # 它定义了数据集的长度，有多少个批次或样本可以用于训练，因为构建数据集时采用的是滑动窗口的方式
        return len(self.data_x) - self.seq_len + 1

    # def inverse_transform(self, data, is_target=True):
    #     # 如果是目标数据 y，则使用目标数据的 scaler 进行反归一化
    #     if is_target:
    #         return self.scaler_y.inverse_transform(data)  # 对目标数据进行反归一化
    #     else:
    #         return self.scaler_x.inverse_transform(data)  # 对输入数据进行反归一化（可以根据需要调整）
    def inverse_transform(self, data, is_target=False):
        # 确保 scaler_y 被正确初始化
        if is_target:
            if self.scaler_y is None:
                print("scaler_y is None, loading from file...")
                scaler_path = r"D:\project\2SOFTS-main\SOFTS-main\scaler_y.pkl"  # 确定路径
                self.scaler_y = joblib.load(scaler_path)  # 加载 scaler_y
                if self.scaler_y is None:
                    raise ValueError("scaler_y could not be loaded.")
                    # 将数据移到 CPU 上
            if isinstance(data, torch.Tensor) and data.is_cuda:
                data = data.cpu().numpy()  # 如果是 GPU 上的 Tensor，将其转移到 CPU 上并转换为 NumPy 数组
                # 如果数据是 3D（batch_size, channels, features），则展平为 2D
            if data.ndim == 3:
                batch_size, channels, features = data.shape
                data = data.reshape(-1, features)  # 展平为 (batch_size * channels, features)

                # 进行反归一化
            data_unnormalized = self.scaler_y.inverse_transform(data)

                # 如果数据是 3D 的，恢复其原来的形状
            if data.ndim == 3:
                data_unnormalized = data_unnormalized.reshape(batch_size, channels, features)
            return data_unnormalized  # 对目标数据进行反归一化
            # return self.scaler_y.inverse_transform(data)  # 对目标数据进行反归一化
        else:
            if self.scaler_x is None:
                print("Error: scaler_x is None")
                raise ValueError("scaler_x is None")
                # 同样处理 scaler_x 的数据
            if isinstance(data, torch.Tensor) and data.is_cuda:
                data = data.cpu().numpy()  # 如果是 GPU 上的 Tensor，将其转移到 CPU 上并转换为 NumPy 数组
                # 如果数据是 3D（batch_size, channels, features），则展平为 2D
            if data.ndim == 3:
                batch_size, channels, features = data.shape
                data = data.reshape(-1, features)  # 展平为 (batch_size * channels, features)

                # 进行反归一化
            data_unnormalized = self.scaler_x.inverse_transform(data)

                # 恢复为原始形状
            if data.ndim == 3:
                data_unnormalized = data_unnormalized.reshape(batch_size, channels, features)

            return data_unnormalized
            # return self.scaler_x.inverse_transform(data)
    # 归一化
    #
    # if self.scale:
    #     self.scaler_y = MinMaxScaler()
    #     train_data_y = self.data_y[border1s[0]:border2s[0]]
    #     self.scaler_y.fit(train_data_y)
    #     data_y = self.scaler_y.transform(self.data_y)
    #
    # else:
    #     data_y = self.data_y
    # def __len__(self):
    #     # 返回数据集的样本数量
    #     return len(self.data_x)  # 假设每个样本对应一个输入特征和目标值
    #
    # def inverse_transform(self, data):
    #     # 如果需要反向变换，针对已归一化的输入数据进行逆变换
    #     return data  # 假设已经归一化，所以直接返回数据

        #对y归一化
        # for i, (x, y) in enumerate(zip(self.data_x, self.data_y)):
        #     print(f"Sample {i}: x_shape={x.shape}, y_shape={y.shape}")

    # def __getitem__(self, index):
    #     # 获取 seq_x 和 seq_y 对应的数据
    #     seq_x = self.data_x[index]  # 取当前索引的输入特征
    #     seq_y = self.data_y[index]  # 取当前索引的目标数据（假设是标量）
    #
    #     # 时间标记可以根据需求进行修改（当前使用零标记）
    #     seq_x_mark = torch.zeros((seq_x.shape[0], 1))  # 根据实际情况修改
    #     seq_y_mark = torch.zeros((seq_y.shape[0], 1))  # 根据实际情况修改
    #
    #     return torch.from_numpy(seq_x).float(), torch.from_numpy(seq_y).float(), seq_x_mark, seq_y_mark

    # s_begin = index
    # s_end = s_begin + self.seq_len
    # r_begin = s_end
    # r_end = r_begin + self.pred_len
    #
    # print(f"Index: {index}, s_begin: {s_begin}, s_end: {s_end}, r_begin: {r_begin}, r_end: {r_end}")
    # print(f"Data lengths: len(data_x)={len(self.data_x)}, len(data_y)={len(self.data_y)}")
    #
    # # 检查边界
    # if r_end > len(self.data_x):
    #     print(f"Adjusting r_end: was {r_end}, set to {len(self.data_x)}")
    #     r_end = len(self.data_x)
    # seq_x = self.data_x[s_begin:s_end]  # 输入序列
    # seq_y = self.data_y[r_begin:r_end]  # 目标序列
    # print(f"seq_x.shape={seq_x.shape}, seq_y.shape={seq_y.shape}")






