from src.data_representation.Examples import Examples
# Throws AssertionError

# Ŷ(t+h|t) = Y(t)
from src.model_prediction.lstm_apply import apply_lstm
from src.model_training.clusters import GenericCluster


def naive_forecast(time_series):
    assert len(time_series) > 0
    return time_series[-1]


# Ŷ(t+h|t) = Y(t+h-T)
# T:= Period of seasonality
def seasonal_naive_forecast(time_series, T=7):
    assert len(time_series) >= T, "Timeseries too short for seasonal naive_forecast"
    return time_series[-T]


def lstm_forecast(ex: Examples):
    train_ex = Examples()
    train_ex.fill_from_snippets(ex.train_data, test_share=0.0)
    test_ex = Examples()
    test_ex.fill_from_snippets(ex.test_data, test_share=1.0)
    apply_lstm(train_ex, test_ex)


def forecast_LSTM_with_cluster(model: GenericCluster, examples: Examples):
    pred_cluster = [[] for _ in range(model.n_clusters)]
    cluster_ex = model.clusters
    pred_label = model.predict(examples)
    for label, snippet in zip(pred_label, examples.test_data):
        pred_cluster[label].append(snippet)
    for l, cluster in enumerate(cluster_ex):
        if len(pred_cluster[l]) > 0:
            test_ex = Examples()
            test_ex.fill_from_snippets(pred_cluster[l])
            print(f"Starting LSTM Training with {len(cluster.train_data)} training examples and "
                  f"{len(pred_cluster[l])} examples to predict")
            apply_lstm(cluster, test_ex)


def forecast_LSTM(examples: Examples):
    train_ex = Examples().fill_from_snippets(examples.train_data)
    test_ex = Examples().fill_from_snippets(examples.test_data)
    apply_lstm(train_ex, test_ex)
