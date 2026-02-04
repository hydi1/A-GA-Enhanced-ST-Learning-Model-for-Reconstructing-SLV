import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # 获取配置中的超参数
        self.seq_len = configs['seq_len']
        print("self.seq_len",self.seq_len)# 输入序列的长度
        self.pred_len = configs['pred_len']  # 预测长度
        self.input_size = configs['input_size']# 每个时间步的输入特征数
        self.hidden_size = configs['hidden_size']  # 隐藏层大小（GRU单元数）
        self.num_layers = configs['num_layers']  # GRU层数
        self.output_size = configs['output_size']  # 输出大小

        # 定义GRU层
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)

        # 定义一个全连接层用于输出
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # 定义dropout层
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        # GRU 处理输入序列
        # x 的 shape 为 (batch_size, seq_len, input_size)

        gru_out, h_n = self.gru(x)  # gru_out 的 shape 为 (batch_size, seq_len, hidden_size)

        #提取所有的时间步
        last_output = gru_out[:, :, :]  # shape: (batch_size, hidden_size)

        # 添加 dropout
        last_output = self.dropout(last_output)

        # 通过全连接层获得最终预测
        output = self.fc(last_output)  # shape: (batch_size, output_size)
        # print('output',output.shape)#(16,12,1)

        return output


