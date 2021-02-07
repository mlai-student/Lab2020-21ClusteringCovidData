import pickle

import torch
from torch import optim, nn

from src.data_representation.Examples import Examples
from src.model_prediction.lstm_model import Forecaster_LSTM
from torch.utils.data import TensorDataset, DataLoader


def apply_lstm(train_ex: Examples, test_ex: Examples):
    # Define hyperparameters
    tmp_snippet = train_ex.train_data[0]
    input_size = 1 #len(tmp_snippet.time_series.shape)
    hidden_layer_size = 50
    num_layers = 1
    max_prediction_length = 1
    epochs = 100
    batch_size = 100
    learning_rate = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    forecaster = Forecaster_LSTM(input_size, hidden_layer_size, num_layers, max_prediction_length).to(device)

    X_train, _, y_train, _ = train_ex.split_examples()
    X_test, _, y_test, _ = test_ex.split_examples()
    t_X_train, t_X_test = torch.Tensor(X_train).unsqueeze(2), torch.Tensor(X_test).unsqueeze(2)
    t_y_train, t_y_test = torch.Tensor(y_train), torch.Tensor(y_test)

    train_dataset = TensorDataset(t_X_train, t_y_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    test_dataset = TensorDataset(t_X_test, t_y_test)
    test_dataloader = DataLoader(test_dataset, X_test.shape[0], shuffle=False)

    optimizer = optim.SGD(forecaster.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()

    #TODO: validation dataset
    #TODO: multiply scaler to output value, if standardized
    for epoch in range(1, epochs+1):
        train_loss = 0
        for train_batch in train_dataloader:
            optimizer.zero_grad()
            X_batch, y_batch = train_batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            pred = forecaster(X_batch)
            loss = criterion(pred.squeeze(1), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(forecaster.parameters(), .5)
            optimizer.step()
            train_loss += loss.item()

        if epoch % 10 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, train_loss))

    print("Start evaluating forecaster")
    with torch.no_grad():
        predictions = []
        targets = []
        test_loss = 0
        for test_batch in test_dataloader:
            X_batch, y_batch = test_batch
            targets.append(y_batch)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = forecaster(X_batch)
            predictions.extend(pred.detach().numpy())
            loss = criterion(pred.squeeze(1), y_batch)
            test_loss += loss.item()

        print("Total test loss accumulates to: ", test_loss)
        return predictions
