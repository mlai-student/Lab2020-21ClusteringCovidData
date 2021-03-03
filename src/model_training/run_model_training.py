import json, logging
import src.model_training.clusters as cl
from src.data_representation.Examples import load_Examples_from_file

def run_model_training_main(main_config, filename_example, filename_model):
    logging.debug("model_training.Run_model_training started main")
    d_class = {"KMedoids": cl.KMedoids, "KMeans": cl.KMeans, "DBSCAN": cl.DBSCAN,
               "TS_KMeans": cl.TS_KMeans, "TS_KShape": cl.TS_KShape}
    try:
        example = load_Examples_from_file(filename_example)
        metric = main_config["model_training_settings"]['metric']
        model_name = main_config["model_training_settings"]['models']
        c_model = d_class[model_name]
        n_clusters = json.loads(main_config["model_training_settings"]["n_clusters"])
        model = c_model(n_clusters, metric=metric).fit(example)
        model.save_model_and_update_overview(main_config, filename_model)
    except Exception as Argument:
        print(Argument)
        logging.error("Could not open file(s)")
    logging.debug("model_training.Run_model_training ended main")
