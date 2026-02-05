import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.QCONVGRU import ConvGRU
from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.input_size = configs['input_size']
        self.hidden_size = configs['hidden_size']
        self.num_layers = configs['num_layers']
        self.output_size = configs['output_size']
        self.use_norm = configs["use_norm"]
        self.qconvGRU = ConvGRU(input_size=(5, 9),
                                input_dim=4 * 33,
                                hidden_dim=64,
                                kernel_size=(3, 3),
                                num_layers=1,
                                dtype=torch.float32,
                                bias=True)

        self.enc_embedding = DataEmbedding_inverted(configs['seq_len'], configs['d_model'], configs['dropout'])

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
        self.activation = nn.ReLU()

        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)

        self.residual = nn.Linear(32,32)

        self.projection = nn.Linear(configs['hidden_size'], 1, bias=True)
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.use_norm:

            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        _, _, N = x_enc.shape

        x_enc=x_enc.reshape(x_enc.shape[0], 12, 4*33,5, 9)

        x_enc_qconv= self.qconvGRU(x_enc)

        x_enc_qconv = x_enc_qconv.reshape(x_enc_qconv.shape[0], 12, -1)

        enc_out1 = self.enc_embedding(x_enc_qconv, x_mark_enc)

        enc_out = enc_out1.permute(0, 2, 1)

        conv_out1 = self.conv1d(enc_out)
        enc_pool= self.max_pool(conv_out1)

        conv_out1 = self.conv2d(enc_pool)
        enc_pool = self.max_pool(conv_out1)
        path1 = self.activation(enc_pool)

        conv_out2 = self.path2conv1(enc_out)

        pooled_out = self.avg_pool(conv_out2)

        conv_out2 = self.path2conv2(pooled_out)
        pooled_out = self.avg_pool(conv_out2)
        path2 = self.activation(pooled_out)
        path1 = path1.permute(0, 2, 1)
        residual = self.residual(path1)
        path1=residual.permute(0, 2, 1)
        gru_input = path2 + path1
        dec_out = self.projection(gru_input)

        if self.use_norm:

            dec_out = dec_out * stdev
            dec_out = dec_out + means

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
