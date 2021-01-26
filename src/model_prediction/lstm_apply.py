import torch
from sklearn.metrics import classification_report
from torch import optim, nn

from src.data_representation.Examples import Examples
from src.model_prediction.lstm_model import Forecaster_LSTM
from torch.utils.data import TensorDataset, DataLoader


def apply_lstm(examples: Examples):
    # Define hyperparameters
    tmp_snippet = examples.train_data[0]
    input_size= len(tmp_snippet.ts.shape)
    hidden_layer_size=100
    num_layers=1
    output_size= len(tmp_snippet.label)
    epochs = 10
    batch_size = 60
    learning_rate = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Forecaster_LSTM(input_size, hidden_layer_size, num_layers, output_size).to(device)

    X_train, X_test, y_train, y_test = examples.split_examples()
    t_X_train, t_X_test = torch.Tensor(X_train), torch.Tensor(X_test)
    t_y_train, t_y_test = torch.Tensor(y_train), torch.Tensor(y_test)

    train_dataset = TensorDataset(t_X_train, t_y_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    test_dataset = TensorDataset(t_X_test, t_y_test)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = 0
        for train_batch in train_dataloader:
            optimizer.zero_grad()
            X_batch, y_batch = train_batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            print(f"Training loss: {loss.item()} in epoch {epoch}")
            print(classification_report(y_batch.cpu().numpy(), pred.cpu().numpy(), digits=4))

    print("Start evaluating model")
    with torch.no_grad():
        preds = []
        targets = []
        for test_batch in test_dataloader:
            optimizer.zero_grad()
            X_batch, y_batch = test_batch
            targets.append(y_batch)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            pred = model(X_batch)
            pred.append(pred.cpu().numpy())

        print(classification_report(preds, targets, digits=4))
