import torch.nn as nn


class Forecaster_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=1, output_size=1):
        super(Forecaster_LSTM).__init__()
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            dropout=.2)

        self.linear = nn.Linear(in_features=hidden_layer_size, out_features=output_size)

    def forward(self, input_seq):
        # input of shape (seq_len, batch, input_size)
        out = self.lstm(input_seq.view(len(input_seq), 1, -1))
        pred = self.linear(out.view(len(input_seq), -1))
        return pred
