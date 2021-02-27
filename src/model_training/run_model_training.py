import json, logging
import src.model_training.clusters as cl
from src.data_representation.Examples import load_Examples_from_file

def run_model_training_main(train_config, filename_example, filename_model):
    logging.debug("model_training.Run_model_training started main")
    d_class = {"KMedoids": cl.KMedoids, "KMeans": cl.KMeans, "DBSCAN": cl.DBSCAN,
               "TS_KMeans": cl.TS_KMeans, "TS_KShape": cl.TS_KShape}
    try:
        example = load_Examples_from_file(filename_example)
        metric = train_config['metric']
        model_name = train_config['models']
        c_model = d_class[model_name]
        n_clusters = json.loads(train_config["n_clusters"])
        model = c_model(n_clusters, metric=metric).fit(example)
        model.save_model(filename_model)
    except Exception as Argument:
        print(Argument)
        logging.error("Could not open file(s)")
    logging.debug("model_training.Run_model_training ended main")
