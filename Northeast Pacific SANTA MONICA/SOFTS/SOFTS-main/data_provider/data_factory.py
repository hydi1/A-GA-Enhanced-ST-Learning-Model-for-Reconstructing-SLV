from data_provider.data_loader import Dataset_Npy
from torch.utils.data import DataLoader
import torch
import numpy as np

data_dict = {

    'ssta':Dataset_Npy,
}


def data_provider(args, flag):
    Data = data_dict[args['data']]
    #如果 args['embed'] 不是 'timeF'，则 timeenc 设置为 0
    #如果 args['embed'] 是 'timeF'，则 timeenc 设置为 1
    timeenc = 0 if args['embed'] != 'timeF' else 1
    # print("Data",Data)#Data <class 'data_provider.data_loader.Dataset_Npy'>
    #在exp_long_term_forecasting  中 flag =train/valid/test,没有pred
    if flag == 'test':
        shuffle_flag = False
        drop_last = False#即使最后一个批次的数据不满，也会保留它
        batch_size = args['batch_size']
        freq = args['freq']
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args['freq']
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args['batch_size']  # bsz for train and valid
        freq = args['freq']
    # print("Data",Data)#Data <class 'data_provider.data_loader.Dataset_Npy'>
    # print("flag",flag)#train，当flag =train时，data_set的Data是data_set 被初始化为所选数据集类的实例，data_set 的具体类型取决于 args['data'] 的值
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

    def collate_fn(batch):
        # 假设 batch 是一个包含元组 (seq_x, seq_y, seq_x_mark, seq_y_mark) 的列表
        seq_x_batch, seq_y_batch, seq_x_mark_batch, seq_y_mark_batch = zip(*batch)

        # 转换为 Tensor，如果数据已经是 Tensor，这个操作是无害的
        seq_x_batch = [torch.tensor(x) if isinstance(x, np.ndarray) else x for x in seq_x_batch]
        seq_y_batch = [torch.tensor(y) if isinstance(y, np.ndarray) else y for y in seq_y_batch]
        seq_x_mark_batch = [torch.tensor(m) if isinstance(m, np.ndarray) else m for m in seq_x_mark_batch]
        seq_y_mark_batch = [torch.tensor(m) if isinstance(m, np.ndarray) else m for m in seq_y_mark_batch]

        # 堆叠 Tensor
        seq_x_batch = torch.stack(seq_x_batch, dim=0)
        seq_y_batch = torch.stack(seq_y_batch, dim=0)
        seq_x_mark_batch = torch.stack(seq_x_mark_batch, dim=0)
        seq_y_mark_batch = torch.stack(seq_y_mark_batch, dim=0)

        return seq_x_batch, seq_y_batch, seq_x_mark_batch, seq_y_mark_batch

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,#32
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last, collate_fn=collate_fn)
    #len(data_loader)=len(data_set)//batch_size+1,即451//32+1=15,即每个epoch有15个batch
    return data_set, data_loader