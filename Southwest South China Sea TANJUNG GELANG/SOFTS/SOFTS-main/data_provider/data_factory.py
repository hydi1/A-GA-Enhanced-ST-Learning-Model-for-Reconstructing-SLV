from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred, Dataset_Random,Dataset_Npy
from torch.utils.data import DataLoader
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'random': Dataset_Random,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'ssta':Dataset_Npy,
}

def data_provider(args, flag):
    Data = data_dict[args['data']]

    timeenc = 0 if args['embed'] != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args['batch_size']
        freq = args['freq']
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args['freq']
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args['batch_size']
        freq = args['freq']

    data_set = Data(
        root_path=args['root_path'],
        data_path=args['data_path'],
        target_path=args['target_path'],
        flag=flag,
        size=[args['seq_len'], args['label_len'], args['pred_len']],
        features=args['features'],
        target=args['target'],
        scale=args['scale'],
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args['seasonal_patterns']
    )

    import numpy as np
    def collate_fn(batch):

        seq_x_batch, seq_y_batch, seq_x_mark_batch, seq_y_mark_batch = zip(*batch)

        seq_x_batch = [torch.tensor(x) if isinstance(x, np.ndarray) else x for x in seq_x_batch]
        seq_y_batch = [torch.tensor(y) if isinstance(y, np.ndarray) else y for y in seq_y_batch]
        seq_x_mark_batch = [torch.tensor(m) if isinstance(m, np.ndarray) else m for m in seq_x_mark_batch]
        seq_y_mark_batch = [torch.tensor(m) if isinstance(m, np.ndarray) else m for m in seq_y_mark_batch]

        seq_x_batch = torch.stack(seq_x_batch, dim=0)
        seq_y_batch = torch.stack(seq_y_batch, dim=0)
        seq_x_mark_batch = torch.stack(seq_x_mark_batch, dim=0)
        seq_y_mark_batch = torch.stack(seq_y_mark_batch, dim=0)

        return seq_x_batch, seq_y_batch, seq_x_mark_batch, seq_y_mark_batch

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last, collate_fn=collate_fn)

    return data_set, data_loader
