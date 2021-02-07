import copy
import pickle

import torch
from sklearn.metrics import accuracy_score
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

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
    batch_size = 50
    learning_rate = 1e-2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    forecaster = Forecaster_LSTM(input_size, hidden_layer_size, num_layers, max_prediction_length).to(device)
    best_forecaster = copy.deepcopy(forecaster)

    X, _, y, _ = train_ex.split_examples()
    split = int(len(X) * .1)
    X_train, X_val, y_train, y_val = X[:split], X[split:], y[:split], y[split:]
    X_test, _, y_test, _ = test_ex.split_examples()
    t_X_train, t_y_train = torch.Tensor(X_train).unsqueeze(2), torch.Tensor(y_train)
    t_X_val, t_y_val = torch.Tensor(X_val).unsqueeze(2), torch.Tensor(y_val)
    t_X_test, t_y_test = torch.Tensor(X_test).unsqueeze(2), torch.Tensor(y_test)

    train_dataset = TensorDataset(t_X_train, t_y_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    val_dataset = TensorDataset(t_X_val, t_y_val)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)

    test_dataset = TensorDataset(t_X_test, t_y_test)
    test_dataloader = DataLoader(test_dataset, X_test.shape[0], shuffle=False)

    optimizer = optim.Adam(forecaster.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, min_lr=1e-5)

    best_loss = np.inf
    for epoch in range(1, epochs+1):
        '''
        Training Routine 
        '''
        train_loss = 0
        for train_batch in train_dataloader:
            optimizer.zero_grad()
            X_batch, y_batch = train_batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            pred = forecaster(X_batch)
            loss = criterion(pred.squeeze(1), y_batch)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(forecaster.parameters(), .5)
            optimizer.step()
            train_loss += loss.item()

        '''
        Validation Routine 
        '''
        val_loss = 0
        for val_batch in val_dataloader:
            optimizer.zero_grad()
            X_batch, y_batch = val_batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            pred = forecaster(X_batch)
            val_loss = criterion(pred.squeeze(1), y_batch)
            scheduler.step(val_loss)
            val_loss += val_loss.item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_forecaster = copy.deepcopy(forecaster)

        if epoch % 10 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, train_loss))

    print("Start evaluating forecaster")
    with torch.no_grad():
        predictions = []
        targets = []
        test_loss = 0
        for val_batch in test_dataloader:
            X_batch, y_batch = val_batch
            targets.append(y_batch)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = best_forecaster(X_batch)
            predictions.extend(pred.detach().numpy())
            loss = criterion(pred.squeeze(1), y_batch)
            test_loss += loss.item()

        print("Total test loss accumulates to: ", test_loss)
        return predictions
