import logging
from src.data_representation.Examples import Examples, load_Examples_from_file
from src.model_training.ts_learn_algorithms import KMeans ,KernelMeans, KNeighbors


#start the model training process with a configuration set given
def run_model_training_main(train_config):
    snippet_set = load_Examples_from_file("data/Dec-02-2020/snippets")
    X_train, X_test, y_train, y_test = snippet_set.make_ts_snippet()
    KMeans(X_train, 20, "euclidean")


    logging.debug("model_training.Run_model_training started main")
    logging.debug("model_training.Run_model_training ended main")
