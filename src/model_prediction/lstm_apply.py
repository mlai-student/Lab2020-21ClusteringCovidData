import copy, torch, logging
import numpy as np
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

from src.data_representation.Examples import Examples
from src.model_prediction.lstm_model import Forecaster_LSTM


def apply_lstm(train_ex: Examples, test_ex: Examples):
    # Define hyperparameters
    tmp_snippet = train_ex.train_data[0]
    input_size, num_layers = 1 ,1
    epochs = 500
    batch_size = 20
    learning_rate = 1e-4
    use_lstm = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.debug(f"For LSTM Training the following device is used: {device}")

    forecaster = Forecaster_LSTM(input_size, 100, num_layers).to(device)
    # forecaster = Forecaster_Simple(len(tmp_snippet.time_series), num_layers=num_layers, layers=[100])
    best_forecaster = copy.deepcopy(forecaster)

    X, _, y, _ = train_ex.split_examples()
    split = int(len(X) * .9)
    #TODO shuffeln ?
    X_train, X_val, y_train, y_val = X[:split], X[split:], y[:split], y[split:]
    X_test, _, y_test, _ = test_ex.split_examples()
    t_X_train, t_y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    t_X_val, t_y_val = torch.Tensor(X_val), torch.Tensor(y_val)
    t_X_test, t_y_test = torch.Tensor(X_test), torch.Tensor(y_test)

    train_dataset = TensorDataset(t_X_train, t_y_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    val_dataset = TensorDataset(t_X_val, t_y_val)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)

    test_dataset = TensorDataset(t_X_test, t_y_test)
    test_dataloader = DataLoader(test_dataset, X_test.shape[0], shuffle=False)

    optimizer = optim.Adam(forecaster.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr=1e-6)

    best_loss = np.inf
    for epoch in range(epochs):
        '''
        Training Routine
        '''
        train_loss = 0
        for train_batch in train_dataloader:
            optimizer.zero_grad()
            X_batch, y_batch = train_batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if use_lstm:
                forecaster.hidden = forecaster.reset_hidden_state(device, X_batch.shape[0])
            # forecaster.lstm.flatten_parameters()
            pred = forecaster(X_batch.transpose(0, 1))
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(forecaster.parameters(), 1.)
            optimizer.step()
            train_loss += loss.item()

        '''
        Validation Routine
        '''
        val_loss = 0
        for val_batch in val_dataloader:
            X_batch, y_batch = val_batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if use_lstm:
                forecaster.hidden = forecaster.reset_hidden_state(device, X_batch.shape[0])
            pred = forecaster(X_batch.transpose(0, 1))
            val_loss = criterion(pred, y_batch)
            scheduler.step(val_loss)
            val_loss += val_loss.item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_forecaster = copy.deepcopy(forecaster)

        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, train_loss))

    print("Start evaluating forecaster")
    with torch.no_grad():
        predictions, targets, test_loss = [], [], 0
        for test_batch in test_dataloader:
            X_batch, y_batch = test_batch
            targets.append(y_batch)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if use_lstm:
                best_forecaster.hidden = best_forecaster.reset_hidden_state(device, X_batch.shape[0])
            pred = best_forecaster(X_batch.transpose(0, 1))
            predictions.extend(pred.cpu().detach().numpy())
            loss = criterion(pred, y_batch)
            test_loss += loss.item()

        print("Total test loss accumulates to: ", test_loss)
        return predictions
