import logging
import pickle

from tslearn.clustering import TimeSeriesKMeans, KernelKMeans
from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.utils import to_time_series_dataset

from src.data_representation.Examples import load_Examples_from_file
import numpy as np


# start the model prediction process with a configuration set given
def run_model_prediction_main(pred_config):
    logging.debug("model_prediction.Run_model_prediction started main")
    with open("data/Dec-03-2020/model/NearestNeighbors", 'rb') as f:
        knn = pickle.load(f)
    # km =TimeSeriesKMeans.from_pickle("data/Dec-02-2020/model/TimeSeriesKMEans")

    X_train, X_test, y_train, y_test = load_Examples_from_file("data/Dec-03-2020/model/NearestNeighbors_data")

    distances, indices = knn.kneighbors(X_test)

    mean_sqrt_error_knn = np.mean([abs(y_test[i] - np.mean([y_train[idx] for idx in x])) \
                                   / y_test[i] if y_test[i] != 0 else 0 for i, x in enumerate(indices)])
    print(mean_sqrt_error_knn)

    logging.debug("model_prediction.Run_model_prediction ended main")