import argparse
import configparser
import logging
import os
from types import SimpleNamespace

import pandas as pd
import json
import itertools
from datetime import date
from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil
from src.data_generation.run_data_generating import run_data_generating_main
from src.model_training.run_model_training import run_model_training_main
from src.model_prediction.run_model_prediction import run_model_prediction_main
from src.data_representation.config_to_dict import get_config_dict

def run_with_std_config():
    print("Run started with standard config")
    main("config.ini")

def run_project_w_unique_config(config, filename_example, filename_model, foldername):
    # run the three main parts of the program flow data generation, model_training and prediciton
    config["main_flow_settings"]["generated_folder_path"] = foldername
    try:
        if config["main_flow_settings"].getboolean("Run_data_generation"):
            run_data_generating_main(config["data_generating_settings"], filename_example)
        config["main_flow_settings"]["generated_data_path"] = filename_example
    except Exception as Argument:
        logging.error("Data generation process failed with the following error message:")
        logging.error(str(Argument))
    try:
        if config["main_flow_settings"].getboolean("Run_model_training"):
            run_model_training_main(config["model_training_settings"], filename_example, filename_model)
            config["main_flow_settings"]["generated_model_path"] = filename_model
    except Exception as Argument:
        logging.error("Model training process failed with the following error message:")
        logging.error(str(Argument))
    try:
        if config["main_flow_settings"].getboolean("Run_model_prediction"):
            run_model_prediction_main(config)
    except Exception as Argument:
        logging.error("Model prediction process failed with the following error message:")
        logging.error(str(Argument))


# read the config file and start the program flow on those information
def main(path_to_cfg):
    # read the config file
    main_config = configparser.ConfigParser()
    main_config.read(path_to_cfg)
    # set logging logging level and remove the old file:
    run_log_filename = "run.log"
    if os.path.exists(run_log_filename):
        os.remove(run_log_filename)
    logging.basicConfig(filename=run_log_filename, level=logging.DEBUG)
    # basis for filenames later on
    today = date.today().strftime("%b-%d-%Y")
    foldername = "data/{}/".format(today)
    #check if folder already exists and throw warning:
    folder_already_exists = os.path.isdir(foldername)
    if folder_already_exists:
        print("Attention the folder already exists")
        #if data generation is supposed to run then remove the folder recursively
        if main_config["main_flow_settings"].getboolean("Run_data_generation"):
            print(f"Run data generation is set to yes so {foldername} gets removed")
            shutil.rmtree(foldername)

    Path(foldername).mkdir(parents=True, exist_ok=True)
    print(f"data folder created under path: {foldername}")
    #copy the config file into the folder to track what has been done there
    shutil.copyfile(path_to_cfg, foldername+"used_config.ini")

    # go through all required cominations defined in config.ini
    # possible variables for combinations:
    lst_divide_by_country_population = json.loads(
        main_config["data_generating_settings"]["divide_by_country_population"])
    lst_do_smoothing = json.loads(main_config["data_generating_settings"]["do_smoothing"])
    lst_nr_days_for_avg = json.loads(main_config["data_generating_settings"]["nr_days_for_avg"])
    # missing choosed method -> still open to implement
    lst_do_data_augmentation = json.loads(main_config["data_generating_settings"]["do_data_augmentation"])
    lst_percent_varianz = json.loads(main_config["data_generating_settings"]["percent_varianz"])
    forecast_evaluation_function = json.loads(main_config["model_prediction_settings"]["forecast_evaluation_function"])
    forecast_function = json.loads(main_config["model_prediction_settings"]["forecast_function"])
    metric = json.loads(main_config["model_training_settings"]["metric"])
    models = json.loads(main_config["model_training_settings"]["models"])
    n_clusters = json.loads(main_config["model_training_settings"]["n_clusters"])
    # copy main config and go through all possible compinations of them above:
    # remove not possible combinations like do_smoothing with a smoothin param
    # run project with adjustet config and save output with given filename_example

    config_comb = list(itertools.product(lst_divide_by_country_population,
                                         lst_do_smoothing, lst_nr_days_for_avg,
                                         lst_do_data_augmentation, lst_percent_varianz, forecast_evaluation_function, forecast_function))

    models_comb = list(itertools.product(metric, models, n_clusters))
    valid_model_combs = {"euclidean": ["KMedoids", "KMeans", "DBSCAN"],
                         "dtw": ["TS_KMeans", "TS_KShape"]}
    # attention: not create not necessary rows
    comb_lists = []
    for i, comb in enumerate(config_comb):
        comb_list = list(comb)
        if comb_list[1] == "no":
            comb_list[2] = -1
        if comb_list[3] == "no":
            comb_list[4] = -1
        if comb_list not in comb_lists:
            comb_lists.append(comb_list)

    if main_config["main_flow_settings"].getboolean("Use_newest_data"):
        main_config["model_training_settings"]["data_path"] = foldername

    print("Run through all combinations")
    print(f"Nr combinations: {len(comb_lists)}")
    for i, comb in tqdm(enumerate(comb_lists)):
        #print("Run project with settings: " + str(comb))
        logging.info("Run project with settings: " + str(comb))
        main_config["data_generating_settings"]["divide_by_country_population"] = str(comb[0])
        main_config["data_generating_settings"]["do_smoothing"] = str(comb[1])
        main_config["data_generating_settings"]["nr_days_for_avg"] = str(comb[2])
        main_config["data_generating_settings"]["do_data_augmentation"] = str(comb[3])
        main_config["data_generating_settings"]["percent_varianz"] = str(comb[4])
        main_config["model_prediction_settings"]["forecast_evaluation_function"] = str(comb[5])
        main_config["model_prediction_settings"]["forecast_function"]  = str(comb[6])

        for m_comb in models_comb:
            if str(m_comb[1]) in valid_model_combs[str(m_comb[0])]:
                print(f"Start training with {m_comb[1]} with metric {m_comb[0]}")
                main_config["model_training_settings"]["metric"] = str(m_comb[0])
                main_config["model_training_settings"]["models"] = str(m_comb[1])
                main_config["model_training_settings"]["n_clusters"] = str(m_comb[2])
                filename_example = foldername + str(i)
                filename_model = f"{str(m_comb[1])}_{hash(filename_example)}_{str(m_comb[2])}"
                run_project_w_unique_config(main_config, filename_example, filename_model, foldername)

                # write entry to the overview.csv
                cfg_settings_dict = get_config_dict(main_config)
                cfg_settings_dict["filename_example"] = filename_example
                cfg_settings_dict["filename_model"] = filename_model
                csv_filename=foldername + "overview.csv"
                if os.path.isfile(csv_filename):
                    df = pd.read_csv(csv_filename)
                    pd_dict = pd.Series(cfg_settings_dict).to_frame().T
                    model_df = df.append(pd_dict.iloc[0])
                else:
                    model_df = pd.Series(cfg_settings_dict).to_frame().T
                model_df.to_csv(csv_filename, index=False)
    print("Done")
    logging.shutdown()
    return 0


# main run entry -> always run this method and use a config file to determine what to do exaclty afterwards
# run it with an config file given: e.g. python3 run.py --path_to_cfg config.ini
if __name__ == '__main__':
    # parse argument: credentials should never be pushed to Git
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_cfg", help="Give a full path under quotes")
    args = parser.parse_args()
    main(path_to_cfg=args.path_to_cfg)
