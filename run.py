import argparse
import configparser
import logging
import os
import pandas as pd
import json
import itertools
from src.data_generation.run_data_generating import run_data_generating_main
from src.model_training.run_model_training import run_model_training_main
from src.model_prediction.run_model_prediction import run_model_prediction_main

def run_with_std_config():
    print("Run started with standart config")
    main("config.ini")


def run_project_w_unique_config(config, filename):
    #run the three main parts of the program flow data generation, model_training and prediciton
    try:
        if config["main_flow_settings"].getboolean("Run_data_generation"):
            run_data_generating_main(config["data_generating_settings"], filename)
    except Exception as Argument:
        logging.error("Data generation process failed with the following error message:")
        logging.error(str(Argument))
    try:
        if config["main_flow_settings"].getboolean("Run_model_training"):
            run_model_training_main(config["model_training_settings"])
    except Exception as Argument:
        logging.error("Model training process failed with the following error message:")
        logging.error(str(Argument))
    try:
        if config["main_flow_settings"].getboolean("Run_model_prediction"):
            run_model_prediction_main(config["model_prediction_settings"])
    except Exception as Argument:
        logging.error("Model prediction process failed with the following error message:")
        logging.error(str(Argument))



#read the config file and start the program flow on those information
def main(path_to_cfg):
    # read the config file
    main_config = configparser.ConfigParser()
    main_config.read(path_to_cfg)
    #set logging logging level and remove the old file:
    run_log_filename = "run.log"
    if os.path.exists(run_log_filename):
        os.remove(run_log_filename)
    logging.basicConfig(filename=run_log_filename, level=logging.DEBUG)

    overview_file = pd.DataFrame([], columns = ["divide_by_country_population", "do_smoothing","nr_days_for_avg", "do_data_augmentation", "percent_varianz","filename"])
    #go through all required cominations defined in config.ini
    #possible variables for combinations:
    lst_divide_by_country_population= json.loads(main_config["data_generating_settings"]["divide_by_country_population"])
    lst_do_smoothing= json.loads(main_config["data_generating_settings"]["do_smoothing"])
    lst_nr_days_for_avg= json.loads(main_config["data_generating_settings"]["nr_days_for_avg"])
    #missing choosed method -> still open to implement
    lst_do_data_augmentation= json.loads(main_config["data_generating_settings"]["do_data_augmentation"])
    lst_percent_varianz= json.loads(main_config["data_generating_settings"]["percent_varianz"])
    #copy main config and go through all possible compinations of them above:
    #remove not possible combinations like do_smoothing with a smoothin param
    #run project with adjustet config and save output with given filename

    config_comb = list(itertools.product(lst_divide_by_country_population,
        lst_do_smoothing,
        lst_nr_days_for_avg,
        lst_do_data_augmentation,
        lst_percent_varianz))
    for i, comb in enumerate(config_comb):
        print("Run project with settings: " + str(comb))
        logging.info("Run project with settings: " + str(comb))
        main_config["data_generating_settings"]["divide_by_country_population"] = str(comb[0])
        main_config["data_generating_settings"]["do_smoothing"] = str(comb[1])
        main_config["data_generating_settings"]["nr_days_for_avg"] = str(comb[2])
        main_config["data_generating_settings"]["do_data_augmentation"] = str(comb[3])
        main_config["data_generating_settings"]["percent_varianz"] = str(comb[4])
        filename = str(i)
        run_project_w_unique_config(main_config, filename)
        new_row = pd.Series(list(comb) +[filename], index = overview_file.columns)
        overview_file = overview_file.append(new_row, ignore_index=True)
        print("Done")
        #TODO discard multiple solutions

    #TODO set to correct location
    overview_file.to_csv("overview.csv")
    logging.shutdown()
    return 0

#main run entry -> always run this method and use a config file to determine what to do exaclty afterwards
#run it with an config file given: e.g. python3 run.py --path_to_cfg config.ini
if __name__ == '__main__':
    # parse argument: credentials should never be pushed to Git
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_cfg", help="Give a full path under quotes")
    args = parser.parse_args()
    main(path_to_cfg=args.path_to_cfg)
