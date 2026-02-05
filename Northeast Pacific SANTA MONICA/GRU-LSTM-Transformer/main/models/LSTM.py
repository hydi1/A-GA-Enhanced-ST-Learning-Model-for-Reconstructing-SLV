import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs['seq_len']

        self.pred_len = configs['pred_len']
        self.input_size = configs['input_size']
        self.hidden_size = configs['hidden_size']
        self.num_layers = configs['num_layers']
        self.output_size = configs['output_size']

        self.lstm = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):

        lstm_out, (h_n, c_n) = self.lstm(x)

        last_output = lstm_out[:, :, :]

        last_output = self.dropout(last_output)

        output = self.fc(last_output)

        return output
