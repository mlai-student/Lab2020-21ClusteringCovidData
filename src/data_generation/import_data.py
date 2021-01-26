import logging

import pandas as pd

def get_ecdc_dataset(data_gen_config):
    df = ""
    try:
        #check whether the data should be imported from a file or a link:
        if data_gen_config.getboolean("use_dataset_from_file"):
            #expecting a pickle file:
            df = pd.read_pickle(data_gen_config["dataset_filename"])
        else:
            df = pd.read_csv(ecdc_url)
        # save the date of interest as datetime object
        df["dateRep"] = pd.to_datetime(df["dateRep"], format='%d/%m/%Y')
    except Exception as Argument:
        logging.error("getting ecdc dataset via url failed with message:")
        logging.error(str(Argument))
    return df
