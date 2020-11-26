import argparse
import configparser
import logging
import os
from src.data_generating.run_data_generating import run_data_generating_main
from src.model_training.run_model_training import run_model_training_main
from src.model_prediction.run_model_prediction import run_model_prediction_main


#read the config file and start the program flow on those information
def main(path_to_cfg):
    # read the config file
    config = configparser.ConfigParser()
    config.read(path_to_cfg)

    #set logging logging level and remove the old file:
    run_log_filename = "run.log"
    if os.path.exists(run_log_filename):
        os.remove(run_log_filename)
    logging.basicConfig(filename=run_log_filename, level=logging.DEBUG)

    #run the three main parts of the program flow data generation, model_training and prediciton
    try:
        if config["main_flow_settings"].getboolean("Run_data_generation"):
            run_data_generating_main(config["data_generating_settings"])
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

    return 0

#main run entry -> always run this method and use a config file to determine what to do exaclty afterwards
#run it with an config file given: e.g. python3 run.py --path_to_cfg config.ini
if __name__ == '__main__':
    # parse argument: credentials should never be pushed to Git
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_cfg", help="Give a full path under quotes")
    args = parser.parse_args()
    main(path_to_cfg=args.path_to_cfg)
