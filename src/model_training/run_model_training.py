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

        models = [cl.KMedoids, cl.KMeans, cl.TS_KMeans, cl.TS_KShape] #train_config["models"]
        n_clusters = train_config["n_clusters"]
        n_clusters = [3,4,5,7,10]

        divide_by_country_population, do_smoothing, nr_days_for_avg, m_filename = [], [], [], []
        do_data_augmentation, percent_varianz, model_name, no_cluster = [], [], [], []

        for n in n_clusters:
            for m in models:
                model = m(n, metric='euclidean').fit(examples)
                model_filename = f"{model.name}_{n}_{hash(filename)}"
                model.save_model(train_config["data_path"], model_filename)

                divide_by_country_population.append(data_config["divide_by_country_population"])
                do_smoothing.append(data_config["do_smoothing"])
                nr_days_for_avg.append(data_config["nr_days_for_avg"])
                do_data_augmentation.append(data_config["do_data_augmentation"])
                percent_varianz.append(data_config["percent_varianz"])
                model_name.append(model.name)
                no_cluster.append(n)
                m_filename.append(model_filename)

        keys = ["divide_by_country_population", "do_smoothing", "nr_days_for_avg",
                "do_data_augmentation", "percent_varianz", "model_name", "no_cluster", "filename"]
        values = [divide_by_country_population, do_smoothing, nr_days_for_avg,
                  do_data_augmentation, percent_varianz, model_name, no_cluster, m_filename]
        model_dict = OrderedDict(zip(keys, values))

        if os.path.isfile(train_config["data_path"] + "models.csv"):
            model_df = pd.read_csv(train_config["data_path"] + "models.csv")
            pd_model_dict = pd.DataFrame.from_dict(model_dict)
            model_df = model_df.append(pd_model_dict, ignore_index=True)
        else:
            print(model_dict)
            model_df = pd.DataFrame.from_dict(model_dict)
        model_df.to_csv(train_config["data_path"] + "models.csv", index=False)

    except Exception as Argument:
        print(Argument)
        logging.error("Could not open file(s)")

    logging.debug("model_training.Run_model_training ended main")

# if train_config.getboolean("save_variance"):
#     keys.append("variance")