import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.CONVGRU import ConvGRU
from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer
from scipy.ndimage import zoom
#STar Aggregate-Redistribute 模块:对输入时间序列数据进行变换和重组


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
        # 普通的convGRU层
        self.convGRU = ConvGRU(input_size=(5, 9),  # 输入特征图的高度和宽度
                               input_dim=4 * 33,  # 输入通道数hidC
                               hidden_dim=64,  # 隐藏状态的通道数hidR,
                               kernel_size=(3, 3),  # 卷积核大小
                               num_layers=1,  # 层数:
                               dtype=torch.float32,  # 数据类型
                               bias=True)
        # Embedding 数据嵌入层
        #DataEmbedding_inverted 是一个嵌入层，将原始输入数据（x_enc）编码为 d_model 维度的高维向量。
        self.enc_embedding = DataEmbedding_inverted(configs['seq_len'], configs['d_model'], configs['dropout'])
        # path1 卷积池化层
        self.conv1d = nn.Conv1d(
            in_channels=configs['d_model'],  # 输入通道数（与嵌入层输出维度一致）
            out_channels=32,  # 输出通道数
            kernel_size=7,  # 卷积核大小（捕捉3个时间步的局部特征）
            padding=3  # 保持序列长度不变
        )
        self.conv2d = nn.Conv1d(
            in_channels=32,  # 输入通道数（与嵌入层输出维度一致）
            out_channels=32,  # 输出通道数
            kernel_size=7,
            padding=3  # 保持序列长度不变
        )
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # path2卷积池化层
        self.path2conv1 = nn.Conv1d(
            in_channels=configs['d_model'],  # 输入通道数（与嵌入层输出维度一致）
            out_channels=32,  # 输出通道数
            kernel_size=11,
            padding=5  # 保持序列长度不变
        )
        self.path2conv2 = nn.Conv1d(
            in_channels=32,  # 输入通道数（与嵌入层输出维度一致）
            out_channels=32,  # 输出通道数
            kernel_size=11,
            padding=5  # 保持序列长度不变
        )
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()  # 激活函数
        # 残差连接层
        self.residual = nn.Linear(32, 32)
        self.projection = nn.Linear(configs['input_size'], 1, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            #去均值: 减去时间序列的均值，使其以零为中心
            #除以标准差: 对序列进行标准化，减小特征的数值范围。
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        _, _, N = x_enc.shape

        # print("x_enc",x_enc.shape)#([16, 12, 7656]) # batch_size,seq_len,features
        # print("x_mark_enc",x_mark_enc.shape)#torch.Size([16, 12, 3])  batch_size,seq_len,features

        ## 对x_enc 进行 qconvgru处理
        x_enc=x_enc.reshape(x_enc.shape[0], 12, 4*33,5, 9)
        x_enc_conv= self.convGRU(x_enc)
        # print("x_enc_conv",x_enc_conv.shape)#[16, 12, 64, 6, 11] batch,seq,hidR,h,w
        x_enc_conv = x_enc_conv.reshape(x_enc_conv.shape[0], 12, -1)  #batch_size,seq_len,features
        # print("x_enc_conv", x_enc_conv.shape)#
        #嵌入:将输入时间序列 x_enc 和时间标记 x_mark_enc 转换为高维表示
        enc_out1 = self.enc_embedding(x_enc_conv, x_mark_enc)#(batch,seq_len,d_model)
        # print("enc_out",enc_out1.shape)#([16, 4227, 48])
        enc_out = enc_out1.permute(0, 2, 1)#([16, 64, 4227])#(batch,seq_len,d_model)
        # print("enc_out",enc_out.shape)#([16, 48, 4227])
        conv_out1 = self.conv1d(enc_out)
        enc_pool= self.max_pool(conv_out1)

        # print("enc_pool",enc_pool.shape)#([16, 32, 2113])

        conv_out1 = self.conv2d(enc_pool)
        enc_pool = self.max_pool(conv_out1)

        enc_pool = self.activation(enc_pool)
        path1 = enc_pool.permute(0, 2, 1)#b,f,s
        # print("path1",path1.shape)#([16, 1056, 12])
        # 一维卷积处理
        conv_out2 = self.path2conv1(enc_out)  # (batch,seq_len,d_model)
        # print("conv_out", conv_out2.shape)  # [16, 64 , 7659]
        pooled_out = self.max_pool(conv_out2)
        # print("pooled_out", pooled_out.shape)  #[16, 64, 3829]
        conv_out2 = self.path2conv2(pooled_out)  # (batch,seq_len,d_model)

        pooled_out = self.max_pool(conv_out2)
        conv_out = self.activation(pooled_out)  # 激活函数
        residual = self.residual(path1)  # [B,  hidden_size,Seq]

        residual=residual.permute(0, 2, 1)

        gru_input = conv_out + residual

        dec_out = self.projection(gru_input)


        if self.use_norm:

            dec_out = dec_out * stdev
            dec_out = dec_out + means
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
