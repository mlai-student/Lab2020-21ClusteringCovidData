import torch.nn as nn
import torch


class Forecaster_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=1):
        super(Forecaster_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers)

        self.linear = nn.Linear(in_features=hidden_layer_size, out_features=1)

    def reset_hidden_state(self, device, bs):
        return (torch.zeros(1, bs, self.hidden_layer_size).to(device),
                torch.zeros(1, bs, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.unsqueeze(2), self.hidden)
        predictions = self.linear(lstm_out[-1])
        return predictions.view(-1)
