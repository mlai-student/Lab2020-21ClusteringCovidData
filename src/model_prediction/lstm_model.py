import torch.nn as nn


class Forecaster_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=1, output_size=1):
        super(Forecaster_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_layer_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            batch_first=True)

        self.linear = nn.Linear(in_features=hidden_layer_size, out_features=output_size)

    def forward(self, input_seq):
        _, (out, _) = self.lstm(input_seq)
        pred = self.linear(out)
        return pred.squeeze(0)
