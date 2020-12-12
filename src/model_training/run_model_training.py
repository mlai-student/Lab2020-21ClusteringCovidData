import logging
from datetime import date
from pathlib import Path
from src.data_representation.Examples import load_Examples_from_file
from src.model_training.ts_learn_algorithms import kMeans, kernelMeans, kNeighbors, skNeighbors
import pickle


# start the model training process with a configuration set given
def run_model_training_main(train_config):
    logging.debug("model_training.Run_model_training started main")

    snippet_set = load_Examples_from_file("data/Dec-12-2020/snippets")
    X_train, X_test, y_train, y_test = snippet_set.split_examples()
    knn = skNeighbors(X_train, 5)
    save_sk_model(knn, [X_train, X_test, y_train, y_test])
    # km = KMeans(X_train, 20, "euclidean")
    # save_time_series_model(km, [X_train, X_test, y_train, y_test])
    # TODO: FIX
    # KernelMeans(X_train, 20, "gak")

    logging.debug("model_training.Run_model_training ended main")


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
