import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.QCONVGRU import ConvGRU
from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer



class Model(nn.Module):
    #初始化函数，嵌入层、编码器和投影层
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']  # 预测长度
        self.input_size = configs['input_size']  # 每个时间步的输入特征数
        self.hidden_size = configs['hidden_size']  # 隐藏层大小（GRU单元数）
        self.num_layers = configs['num_layers']  # GRU层数
        self.output_size = configs['output_size']  # 输出大小
        self.use_norm = configs["use_norm"]
        self.qconvGRU = ConvGRU(input_size=(6, 11),  # 输入特征图的高度和宽度
                                input_dim=4 * 29,  # 输入通道数hidC
                                hidden_dim=64,  # 隐藏状态的通道数hidR,
                                kernel_size=(3, 3),  # 卷积核大小
                                num_layers=1,  # 层数:
                                dtype=torch.float32,  # 数据类型
                                bias=True)
        self.enc_embedding = DataEmbedding_inverted(configs['seq_len'], configs['d_model'], configs['dropout'])
        # path1 卷积池化层
        self.conv1d = nn.Conv1d(
            in_channels=configs['d_model'],
            out_channels=32,
            kernel_size=7,
            padding=3
        )
        self.conv2d = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=7,
            padding=3
        )
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # path2卷积池化层
        self.path2conv1 = nn.Conv1d(
            in_channels=configs['d_model'],
            out_channels=32,
            kernel_size=11,
            padding=5
        )
        self.path2conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=11,
            padding=5
        )
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()  # 激活函数
        # 定义GRU层
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)
        # 残差连接层
        self.residual = nn.Linear(32,32)
        self.projection = nn.Linear(configs['hidden_size'], 1, bias=True)
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        _, _, N = x_enc.shape

        x_enc=x_enc.reshape(x_enc.shape[0], 12, 4*29,6, 11)#(16, 12, 4* 29, 6, 11)
        x_enc_qconv= self.qconvGRU(x_enc)
        x_enc_qconv = x_enc_qconv.reshape(x_enc_qconv.shape[0], 12, -1)
        enc_out1 = self.enc_embedding(x_enc_qconv, x_mark_enc)#(batch,seq_len,d_model)
        enc_out = enc_out1.permute(0, 2, 1)
        #  path1
        conv_out1 = self.conv1d(enc_out)
        enc_pool= self.max_pool(conv_out1)
        # print("enc_pool",enc_pool.shape)#([16, 32, 2113])
        conv_out1 = self.conv2d(enc_pool)
        enc_pool = self.max_pool(conv_out1)
        path1 = self.activation(enc_pool)

        # print("path1",path1.shape)#([16, 1056, 12])
        # path2
        conv_out2 = self.path2conv1(enc_out)  # (batch,seq_len,d_model)
        # print("conv_out", conv_out2.shape)  # [16, 64 , 7659]
        pooled_out = self.avg_pool(conv_out2)
        # print("pooled_out", pooled_out.shape)  #[16, 64, 3829]
        conv_out2 = self.path2conv2(pooled_out)  # (batch,seq_len,d_model)
        # print("conv_out", conv_out2.shape)  # [16, 64 , 7659]
        pooled_out = self.avg_pool(conv_out2)
        # print("pooled_out", pooled_out.shape)  #([16, 32, 1056])
        path2 = self.activation(pooled_out)  # 激活函数
        # 残差连接：将卷积输出与原始嵌入输出融合
        path1 = path1.permute(0, 2, 1)#b,f,s
        residual = self.residual(path1)  # [B,  hidden_size,Seq]
        path1=residual.permute(0, 2, 1)
        dec_out = path2 + path1

        if self.use_norm:
            dec_out = dec_out * stdev
            dec_out = dec_out + means
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.seq_len:, :]  # [B, L, D]
