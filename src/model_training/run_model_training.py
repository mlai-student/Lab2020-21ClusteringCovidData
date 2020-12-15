import logging
from datetime import date
from pathlib import Path

from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids

from src.data_representation.Examples import load_Examples_from_file
from src.model_training.ts_learn_algorithms import kMeans, kernelMeans, kNeighbors, skNeighbors
import pickle


# start the model training process with a configuration set given
# examples can be loaded by file or transferred as parameter
# models can be defined via config.ini or directly as paramter
# 'auto' clustering first performs DBSCAN to define a clustering size.
# First cluster size greater then 1 is used.
def run_model_training_main(train_config, examples=None, models=None, cluster=None, no_clusters=None):
    logging.debug("model_training.Run_model_training started main")
    if examples is None:
        examples = load_Examples_from_file("data/Dec-12-2020/snippets")
    X_train, X_test, y_train, y_test = examples.split_examples()

    if models is None:
        models = train_config['models']

    distance_matrix = examples.to_distance_matrix()

    if cluster == 'auto':
        for eps in [0.5,1,2,3,4,5,6,7,8,9]:
            db = DBSCAN(eps=eps, metric='precomputed').fit(distance_matrix)
            n_clusters = len(db.core_sample_indices_)
            if n_clusters > 1:
                logging.info(f"Auto clustering found {n_clusters} cluster with eps = {eps} ")
                break
        if n_clusters <=1:
            logging.warning('Could not find automatic cluster size. Continue with default=5 clusters')
            n_clusters = 5

    for model in models:
        if model == 'knn':
            knn = skNeighbors(X_train, 5)
            save_sk_model(knn, [X_train, X_test, y_train, y_test])
        elif model == 'kmeans':
            km = KMeans(X_train, 20, "euclidean")
            save_sk_model(km, [X_train, X_test, y_train, y_test])
        elif model == 'kmedoids':
            kmedoids = KMedoids(n_clusters=13, metric='precomputed', random_state=0).fit(distance_matrix)
            save_sk_model(kmedoids, [X_train, X_test, y_train, y_test])
    logging.debug("model_training.Run_model_training ended main")

# Saving skLearn Model
def save_sk_model(model, examples):
    try:
        today = date.today().strftime("%b-%d-%Y")
        Path("data/" + today).mkdir(parents=True, exist_ok=True)
        Path("data/{}/{}".format(today, "model")).mkdir(parents=True, exist_ok=True)
        with open("data/{}/{}/{}".format(today, "model", str(model.__class__.__name__)), 'wb') as f:
            pickle.dump(model, f)
        with open("data/{}/{}/{}_{}".format(today, "model", str(model.__class__.__name__), "data"), "wb") as pkl_file:
            pickle.dump(examples, pkl_file)

    except Exception as Argument:
        logging.error("Saving model file failed with following message:")
        logging.error(str(Argument))


def save_time_series_model(model, examples):
    try:
        today = date.today().strftime("%b-%d-%Y")
        Path("data/" + today).mkdir(parents=True, exist_ok=True)
        Path("data/{}/{}".format(today, "model")).mkdir(parents=True, exist_ok=True)
        model.to_pickle("data/{}/{}/{}".format(today, "model", str(model.__class__.__name__)))
        with open("data/{}/{}/{}_{}".format(today, "model", str(model.__class__.__name__), "data"), "wb") as pkl_file:
            pickle.dump(examples, pkl_file)

    except Exception as Argument:
        logging.error("Saving model file failed with following message:")
        logging.error(str(Argument))
