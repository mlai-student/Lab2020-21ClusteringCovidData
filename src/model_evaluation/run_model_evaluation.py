import logging
import pickle
from src.data_representation.Examples import load_Examples_from_file
import numpy as np

def run_model_prediction_main(pred_config):
    logging.debug("model_evaluation.Run_model_evaluation started main")
    with open("data/Dec-03-2020/model/NearestNeighbors", 'rb') as f:
        knn = pickle.load(f)

    logging.debug("model_prediction.Run_model_evaluation ended main")