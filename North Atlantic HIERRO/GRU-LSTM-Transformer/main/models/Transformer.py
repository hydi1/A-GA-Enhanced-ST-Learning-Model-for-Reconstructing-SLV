import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs['seq_len']
        print("self.seq_len", self.seq_len)
        self.pred_len = configs['pred_len']
        self.input_size = configs['input_size']
        self.d_model = configs.get('d_model', 64)
        self.nhead = configs.get('nhead', 4)
        self.num_layers = configs.get('num_layers', 2)
        self.output_size = configs['output_size']
        self.dropout = configs.get('dropout', 0.3)

        self.input_projection = nn.Linear(self.input_size, self.d_model)

        self.pos_encoder = PositionalEncoding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        self.fc = nn.Linear(self.d_model, self.output_size)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):

        x = self.input_projection(x)

        x = self.pos_encoder(x)

        output = self.transformer_encoder(x)

        output = self.dropout(output)

        output = self.fc(output)
        print('output', output.shape)

        return output
