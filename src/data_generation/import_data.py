import logging
import pandas as pd

def get_ecdc_dataset(fix_cfg):
    try:
        #check whether the data should be imported from a file or a link:
        if fix_cfg["general_settings"].getboolean("use_dataset_from_file"):
            #expecting a pickle file:
            df = pd.read_pickle(fix_cfg["general_settings"]["dataset_filename"])
        else:
            df = pd.read_csv(ecdc_url)
        # save the date of interest as datetime object
        df["dateRep"] = pd.to_datetime(df["dateRep"], format='%d/%m/%Y')
    except Exception as Argument:
        logging.error("getting ecdc dataset via url failed with message:")
        logging.error(str(Argument))
    return df

def get_standard_ecdc_dataset():
    try:
        df = pd.read_pickle("ecdc_df")
        df["dateRep"] = pd.to_datetime(df["dateRep"], format='%d/%m/%Y')
    except Exception as Argument:
        logging.error("getting ecdc dataset via url failed with message:")
        logging.error(str(Argument))
    return df
