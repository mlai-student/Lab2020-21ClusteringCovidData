import pandas as pd
import numpy as np
import logging
import random
from src.data_representation import Snippet

#start the data generating process with a configuration set given
def run_data_generating_main(data_gen_config):
    logging.debug("data_generating.Run_data_generating started main")
    #first get the raw data from ecdc csv using the url from config file
    df = get_ecdc_dataset(data_gen_config["ecdc_dataset_url"])
    #TODO replace/fill NaN and resize cases but only after analysis beforehand -> control by config ini
    df = df[~(df["popData2019"].isnull())]
    df["cases_per_pop"] = df["cases"] / df["popData2019"] * 100
    df.fillna(0, inplace=True)
    print(df.head())

    #TODO subdivide the ecdc data into snippets using the setting in the config ini
    #snippets = divide_ecdc_data_into_snipptets(ecdc_raw_data, data_gen_config)

    #TODO save snippets and raw_data into data folder!


    logging.debug("data_generating.Run_data_generating finished main")


def get_ecdc_dataset(ecdc_url):
    df = ""
    try:
        df=  pd.read_csv(ecdc_url)
        #save the date of interest as datetime object
        df["dateRep"] = pd.to_datetime(df["dateRep"], format='%d/%m/%Y')

    except Exception as Argument:
        logging.error("getting ecdc dataset via url failed with message:")
        logging.error(str(Argument))
    return df


def divide_ecdc_data_into_snipptets(ecdc_df, data_gen_config):
    data_gen_config["ecdc_dataset_url"]
    snippets = []
    search_val = data_gen_config["search_val"]
    label_length = data_gen_config['label_length']
    length = data_gen_config['snippet_length']
    groups = ecdc_df[['dateRep', data_gen_config["group_by"], search_val]].groupby(data_gen_config["group_by"])
    try:
        # extract Examples from data
        for group in groups:
            group_sort = group[1].sort_values(by='dateRep', ascending=True)
            max_idx = group_sort.shape[0]-length-label_length-1
            indices = make_interval_indices(data_gen_config['overlap'], \
                                            length, data_gen_config['no_snippets'], max_idx)
            for [start, end] in indices:
                X = group_sort.iloc[start : end]
                Y = group_sort.iloc[end+1 : end+1+label_length]
                snippets.append(Snippet(np.array(X[search_val]), np.array(Y[search_val])))

    except Exception as Argument:
        logging.error("converting dataset into snippets failed with message:")
        logging.error(str(Argument))

    return snippets



def make_interval_indices(length, no_intervals, max_idx):
    start_idxs = [random.randint(0, max_idx) for _ in range(no_intervals)]
    return [[x, x+length] for x in start_idxs]

