import torch.nn as nn
import torch


# implementation of a Neural net as simple altertanative to LSTM for cluster and general forecasting
class Forecaster_Simple(nn.Module):
    def __init__(self, input_size=1, num_layers=1, layers=[100]):
        super(Forecaster_Simple, self).__init__()

        assert (num_layers == len(layers))

        self.classifier = nn.ModuleList()
        previous_layer = layers[0]
        first_layer = nn.Linear(input_size, previous_layer)
        torch.nn.init.normal_(first_layer.weight, std=0.02)
        self.classifier.append(first_layer)
        for n_layer in layers[1:]:
            layer = nn.Linear(previous_layer, n_layer)
            torch.nn.init.normal_(layer.weight, std=0.02)
            self.classifier.append(layer)
            previous_layer = n_layer
        last_layer = nn.Linear(previous_layer, out_features=1)
        self.classifier.append(last_layer)

    def forward(self, input_seq):
        out = input_seq.transpose(0, 1)
        for layer in self.classifier:
            out = layer(out)
            out = torch.sigmoid(out)
        return out.view(-1)
