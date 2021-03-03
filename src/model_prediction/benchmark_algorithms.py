from src.data_representation.Examples import Examples
from src.model_prediction.forecast_evaluation_functions import avg_perc_dist
from src.model_prediction.lstm_apply import apply_lstm
from src.model_training.clusters import GenericCluster
import logging
import numpy as np


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
    train_ex.fill_from_snippets(ex.train_data)
    test_ex = Examples()
    test_ex.fill_from_snippets(ex.test_data)
    predictions = apply_lstm(train_ex, test_ex)
    '''Transfering prediction to original snippet'''
    for snippet, pred in zip(ex.test_data, predictions):
        snippet.forecast = pred


def lstm_forecast_cluster(model: GenericCluster, examples: Examples):
    pred_cluster = [[] for _ in range(model.n_clusters)]
    cluster_ex = model.clusters
    pred_label = model.predict(examples)
    for idx, (label, snippet) in enumerate(zip(pred_label, examples.test_data)):
        pred_cluster[label].append([snippet, idx])
    for l, cluster in enumerate(cluster_ex):
        if len(pred_cluster[l]) > 0:
            test_snippets = [p[0] for p in pred_cluster[l]]
            rev_idx = [p[1] for p in pred_cluster[l]]
            test_ex = Examples()
            test_ex.fill_from_snippets(test_snippets)

            print(f"\n\nStarting LSTM Training with {len(cluster.train_data)} training examples and "
                  f"{len(pred_cluster[l])} examples to predict\n")
            predictions, targets = apply_lstm(cluster, test_ex)

            '''Transfering prediction to original snippet'''
            tmp_snippets = []
            for idx, pred, y in zip(rev_idx, predictions, targets):
                snippet = examples.test_data[idx]
                # if y != snippet.label:
                #     print(f"label : {snippet.label} and lstm label: {y} and forecast {pred}")
                snippet.forecast = pred
                tmp_snippets.append(snippet)
            print(f"Abweichung: {avg_perc_dist(tmp_snippets) * 100}% in cluster {l}")

#use the cluster average label to forecast a test snippet
def cluster_avg_forecast(model: GenericCluster, examples: Examples):
    pred_cluster = [[] for _ in range(model.n_clusters)]
    cluster_ex, pred_label = model.clusters, model.predict(examples)

    for (label, snippet) in zip(pred_label, examples.test_data):
        pred_cluster[label].append(snippet)
    for l, cluster in enumerate(cluster_ex):
        #get average cluster label
        prediction = np.mean([snippet.label for snippet in cluster.train_data])
        for snippet in pred_cluster[l]:
            snippet.forecast = prediction
