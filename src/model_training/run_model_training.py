import json
import logging
import os

from src.data_representation.Examples import load_Examples_from_file
import src.model_training.clusters as cl
import pandas as pd
from collections import OrderedDict


def run_model_training_main(train_config, data_config, filename):
    logging.debug("model_training.Run_model_training started main")
    try:
        examples = load_Examples_from_file(filename)
        examples.add_padding()

        if train_config["metric"] == "euclidean":
            models = [cl.KMeans]
            metric = "euclidean"
        elif train_config["metric"] == "dtw":
            models = [cl.TS_KMeans] #cl.TS_KernelKMeans, cl.KMedoids
            metric = "dtw"
        else:
            raise TypeError("Unknown metric")

        n_clusters = json.loads(train_config["n_clusters"])
        divide_by_country_population, do_smoothing, nr_days_for_avg, m_filename = [], [], [], []
        do_data_augmentation, percent_varianz, model_name, no_cluster, metrics = [], [], [], [], []

        keys = ["divide_by_country_population", "do_smoothing", "nr_days_for_avg",
                "do_data_augmentation", "percent_varianz", "filename", "model_name", "no_cluster", "metric"]
        values = [divide_by_country_population, do_smoothing, nr_days_for_avg,
                  do_data_augmentation, percent_varianz, m_filename, model_name, no_cluster, metrics]

        for n in n_clusters:
            for m in models:
                model = m(n, metric=metric).fit(examples)
                model_filename = f"{model.name}_{n}_{hash(filename)}"
                model.save_model(train_config["data_path"], model_filename)
                for k, v in zip(keys[:-4], values):
                    v.append(data_config[k])
                model_name.append(model.name)
                no_cluster.append(n)
                metrics.append(model.metric)
                m_filename.append(model_filename)
        model_dict = OrderedDict(zip(keys, values))

        if os.path.isfile(train_config["data_path"] + "models.csv"):
            model_df = pd.read_csv(train_config["data_path"] + "models.csv")
            pd_model_dict = pd.DataFrame.from_dict(model_dict)
            model_df = model_df.append(pd_model_dict, ignore_index=True)
        else:
            model_df = pd.DataFrame.from_dict(model_dict)
        model_df.to_csv(train_config["data_path"] + "models.csv", index=False)

    except Exception as Argument:
        print(Argument)
        logging.error("Could not open file(s)")

    logging.debug("model_training.Run_model_training ended main")