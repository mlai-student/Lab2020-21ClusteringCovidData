import argparse, configparser
import logging, os, shutil, json, itertools
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data_generation.run_data_generating import run_data_generating_main
from src.model_training.run_model_training import run_model_training_main
from src.model_prediction.run_model_prediction import run_model_prediction_main
from src.data_representation.config_to_dict import get_config_dict


def run_project_w_unique_config(main_cfg, fix_cfg, filename_example, filename_model, foldername):
    # run the three main parts of the program flow data generation, model_training and prediciton
    main_cfg["data_generating_settings"]["generated_folder_path"] = foldername
    try:
        if fix_cfg["main_flow_settings"].getboolean("Run_data_generation"):
            main_cfg["data_generating_settings"]["generated_data_path"] = filename_example
            run_data_generating_main(main_cfg["data_generating_settings"], fix_cfg, filename_example)

    except Exception as Argument:
        logging.error("Data generation process failed with the following error message:")
        logging.error(str(Argument))
    try:
        if fix_cfg["main_flow_settings"].getboolean("Run_model_training"):
            main_cfg["data_generating_settings"]["generated_model_path"] = filename_model
            run_model_training_main(main_cfg, filename_example, filename_model)

    except Exception as Argument:
        logging.error("Model training process failed with the following error message:")
        logging.error(str(Argument))
    try:
        if fix_cfg["main_flow_settings"].getboolean("Run_model_prediction"):
            run_model_prediction_main(main_cfg)
    except Exception as Argument:
        logging.error("Model prediction process failed with the following error message:")
        logging.error(str(Argument))


# read the ini file
def get_ini_arguments(filename):
    main_config = configparser.ConfigParser()
    main_config.read(filename)
    return main_config


# set logging logging level and remove the old file:
def set_logging_settings(log_filename):
    if os.path.exists(log_filename):
        os.remove(log_filename)
    logging.basicConfig(filename=log_filename, level=logging.DEBUG)


# create new folder (if wanted)
def set_output_folder(main_config, fix_cfg, path_to_cfg):
    auto_generate = main_config['data_generating_settings'].getboolean('generate_automatic')
    if auto_generate:
        folder = date.today().strftime("%b-%d-%Y")
    else:
        folder = main_config['data_generating_settings']['generated_folder_name']
    foldername = "data/{}/".format(folder)
    # check if folder already exists and throw warning:
    folder_already_exists = os.path.isdir(foldername)
    if folder_already_exists:
        print("Attention the folder already exists")
        # if data generation is supposed to run then remove the folder recursively
        if fix_cfg["main_flow_settings"].getboolean("Run_data_generation"):
            print(f"Run data generation is set to yes so {foldername} gets removed")
            shutil.rmtree(foldername)

    Path(foldername).mkdir(parents=True, exist_ok=True)
    Path(foldername + "/models/").mkdir(parents=True, exist_ok=True)
    Path(foldername + "/examples_sets/").mkdir(parents=True, exist_ok=True)
    print(f"data folder created under path: {foldername}")
    # copy the config file into the folder to track what has been done there
    shutil.copyfile(path_to_cfg, foldername + "used_config.ini")
    # save the foldername in config
    main_config["model_training_settings"]["data_path"] = foldername
    return foldername


# the config file dictates many different runs through our pipeline return here all possible combinations:
def get_all_valid_cfg_combinations(main_config):
    lst_divide_by_country_population = json.loads(
        main_config["data_generating_settings"]["divide_by_country_population"])
    lst_do_smoothing = json.loads(main_config["data_generating_settings"]["do_smoothing"])
    lst_nr_days_for_avg = json.loads(main_config["data_generating_settings"]["nr_days_for_avg"])
    lst_do_data_augmentation = json.loads(main_config["data_generating_settings"]["do_data_augmentation"])
    lst_percent_varianz = json.loads(main_config["data_generating_settings"]["percent_varianz"])
    forecast_evaluation_function = json.loads(main_config["model_prediction_settings"]["forecast_evaluation_function"])
    forecast_function = json.loads(main_config["model_prediction_settings"]["forecast_function"])
    metric = json.loads(main_config["model_training_settings"]["metric"])
    models = json.loads(main_config["model_training_settings"]["models"])
    n_clusters = json.loads(main_config["model_training_settings"]["n_clusters"])

    config_comb = list(itertools.product(lst_divide_by_country_population, lst_do_smoothing, lst_nr_days_for_avg,
                                         lst_do_data_augmentation, lst_percent_varianz, forecast_evaluation_function,
                                         forecast_function,
                                         metric, models, n_clusters))
    # attention: not create not necessary rows
    # create a list of only important combinations
    comb_lists = []
    for i, comb in enumerate(config_comb):
        comb_list = list(comb)
        if comb_list[1] == "no":
            comb_list[2] = -1
        if comb_list[3] == "no":
            comb_list[4] = -1
        if is_a_valid_comb_list(comb) and comb_list not in comb_lists:
            comb_lists.append(comb_list)
    return comb_lists


# check if a combination is valid or not -> for metric, models, n_clusters
def is_a_valid_comb_list(comb_list):
    valid_model_combs = {"euclidean": ["KMedoids", "KMeans", "DBSCAN"],
                         "dtw": ["TS_KMeans", "TS_KShape"]}
    if str(comb_list[-2]) in valid_model_combs[str(comb_list[-3])]:
        return True
    return False


def add_entry_to_overview_csv(main_config, filename_example, filename_model, foldername):
    # write entry to the overview.csv
    cfg_settings_dict = get_config_dict(main_config)
    cfg_settings_dict["filename_example"] = filename_example
    cfg_settings_dict["filename_model"] = filename_model
    csv_filename = foldername + "overview.csv"
    if os.path.isfile(csv_filename):
        df = pd.read_csv(csv_filename)
        pd_dict = pd.Series(cfg_settings_dict).to_frame().T
        model_df = df.append(pd_dict.iloc[0])
    else:
        model_df = pd.Series(cfg_settings_dict).to_frame().T
    model_df.to_csv(csv_filename, index=False)


def set_main_cfg_to_comb_data(main_config, comb):
    main_config["data_generating_settings"]["divide_by_country_population"] = str(comb[0])
    main_config["data_generating_settings"]["do_smoothing"] = str(comb[1])
    main_config["data_generating_settings"]["nr_days_for_avg"] = str(comb[2])
    main_config["data_generating_settings"]["do_data_augmentation"] = str(comb[3])
    main_config["data_generating_settings"]["percent_varianz"] = str(comb[4])
    main_config["model_prediction_settings"]["forecast_evaluation_function"] = str(comb[5])
    main_config["model_prediction_settings"]["forecast_function"] = str(comb[6])
    main_config["model_training_settings"]["metric"] = str(comb[7])
    main_config["model_training_settings"]["models"] = str(comb[8])
    main_config["model_training_settings"]["n_clusters"] = str(comb[9])


# read the config file and start the program flow on those information
def main(path_to_cfg):
    main_config, variable_config = get_ini_arguments(path_to_cfg), get_ini_arguments("variables_cfg.ini")
    set_logging_settings(variable_config["general_settings"]["log_filename"])
    foldername = set_output_folder(main_config, variable_config, path_to_cfg)
    # get all combinations out of the config file to iterate through:
    comb_lists = get_all_valid_cfg_combinations(main_config)
    print("Run through all combinations")
    print(f"Nr combinations: {len(comb_lists)}")
    for i, comb in tqdm(enumerate(comb_lists)):
        logging.info("Run project with settings: " + str(comb))
        set_main_cfg_to_comb_data(main_config, comb)
        filename_example = foldername + "examples_sets/example_set_" + str(i)
        filename_model = foldername + "models/trained_model_" + str(i)
        run_project_w_unique_config(main_config, variable_config, filename_example, filename_model, foldername)
        add_entry_to_overview_csv(main_config, filename_example, filename_model, foldername)
    print("Done")
    logging.shutdown()
    return 0


# main run entry -> always run this method and use a config file to determine what to do exaclty afterwards
# run it with an config file given: e.g. python3 run.py --path_to_cfg config.ini
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_cfg", help="Give a path for the config file")
    args = parser.parse_args()
    main(path_to_cfg=args.path_to_cfg)
