import logging
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans
from tslearn.neighbors import KNeighborsTimeSeries
from src.data_representation.Examples import load_Examples_from_file
import numpy as np


# start the model prediction process with a configuration set given
def run_model_prediction_main(pred_config):
    logging.debug("model_prediction.Run_model_prediction started main")
    knn: KNeighborsTimeSeries = KNeighborsTimeSeries.from_pickle("data/Dec-02-2020/model/KNeighborsTimeSeries")
    # km =TimeSeriesKMeans.from_pickle("data/Dec-02-2020/model/TimeSeriesKMEans")
    X_train, X_test, y_train, y_test = load_Examples_from_file("data/Dec-02-2020/model/KNeighborsTimeSeries_data")
    #ERROR:root:This KNeighborsTimeSeries instance is not fitted yet.
    #Call 'fit' with appropriate arguments before using this estimator.
    #Ja also ich verstehe es noch nicht
    ind = knn.kneighbors_graph()

    logging.debug("model_prediction.Run_model_prediction ended main")

# from sklearn.neighbors import NearestNeighbors

# def avg_NN(test_set, train_set):
#     test_set_x = list(zip(*test_set))[0]
#     test_set_y = list(zip(*test_set))[1]
#     train_set_x =list(zip(*train_set))[0]
#     train_set_y =  list(zip(*train_set))[1]
#     means = []
#     xs = list(range(1,120))
#     for i in xs:
#         nbrs = NearestNeighbors(n_neighbors=i, algorithm='auto').fit(train_set_x)
#         distances, indices = nbrs.kneighbors(test_set_x)
#         test_set_y, 
#         means.append(np.mean([abs(test_set_y[i] - np.mean([train_set_y[indice] for indice in x]))/test_set_y[i] if test_set_y[i] != 0 else 0 for i in indices]))
#     return xs, means
