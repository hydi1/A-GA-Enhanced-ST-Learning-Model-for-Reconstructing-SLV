import torch
import torch.nn as nn

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):

        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.conv3d= nn.Conv3d(in_channels=12, out_channels=12, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.pool3d_layer = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
    def forward(self, x, x_mark):

        x = x.permute(0, 2, 1)

        if x_mark is None:
            x = self.value_embedding(x)
        else:

            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))

        return self.dropout(x)
