import logging
from src.data_representation.Examples import load_Examples_from_file
import src.model_training.clusters as cl
import pandas as pd

def run_model_training_main(overview_filename, train_config):
    logging.debug("model_training.Run_model_training started main")
    try:
        overview_df = pd.read_csv(train_config["data_path"] + overview_filename)
        data = []
        for f in overview_df['filename']:
            data.append((load_Examples_from_file(train_config["data_path"] + str(f)), f))
        models = train_config["models"]
        n_clusters = train_config["n_clusters"]
        for n in n_clusters:
            for m in models:
                model_filenames = []
                model = None
                for ex, file in data:
                    ex.add_padding()
                    model = m(n, metric='euclidean').fit(ex)
                    filename = f"{model.name}_{n}_{file}"
                    model.save_model(filename)
                    model_filenames.append(filename)
                overview_df[f"{model.name}_{n}_file"] = model_filenames
    except Exception as Argument:
        logging.error("Could not open file(s)")

    logging.debug("model_training.Run_model_training ended main")