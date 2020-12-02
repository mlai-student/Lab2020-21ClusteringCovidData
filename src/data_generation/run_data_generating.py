import pandas as pd
import numpy as np
import logging
import random
from datetime import date
from pathlib import Path
from src.data_representation.Snippet import Snippet
from src.data_representation.Examples import Examples

#start the data generating process with a configuration set given
def run_data_generating_main(data_gen_config):
    logging.debug("data_generating.Run_data_generating started main")
    #first get the raw data from ecdc csv using the url from config file
    df = get_ecdc_dataset(data_gen_config["ecdc_dataset_url"])
    #TODO replace/fill NaN and resize cases but only after analysis beforehand -> control by config ini
    df = df[~(df["popData2019"].isnull())]
    df["cases_per_pop"] = df["cases"] / df["popData2019"] * 100
    df.fillna(0, inplace=True)
    #print(df.head())
    save_data_frame(df)

    snippets = divide_ecdc_data_into_snippets(df, data_gen_config)
    snippet_Examples = Examples()
    snippet_Examples.fill_from_snippets(snippets)
    snippet_Examples.save_to_file("snippets")

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


def save_data_frame(df):
    try:
        today = date.today().strftime("%b-%d-%Y")
        Path("data/" + today).mkdir(parents=True, exist_ok=True)
        df.to_pickle("data/{}/{}".format(today, "ecdc_df"))
    except Exception as Argument:
        logging.error("Saving dataframe failed with following message:")
        logging.error(str(Argument))


def divide_ecdc_data_into_snippets(ecdc_df, data_gen_config):
    snippets = []
    search_val = data_gen_config["examples_search_val"]
    group_by = data_gen_config["examples_group_by"]
    try:
        snippet_length = int(data_gen_config['examples_snippet_length'])
        label_length = int(data_gen_config['examples_label_length'])
    except ValueError as Argument:
        logging.error("Converting config to string failed with message:")
        logging.error(str(Argument))
    try:
        # extract Examples from data
        groups = ecdc_df[['dateRep', group_by, search_val]].groupby(group_by)
        for group in groups:
            group_sort = group[1].sort_values(by='dateRep', ascending=True)
            max_idx = group_sort.shape[0]-snippet_length-label_length-1
            indices = make_interval_indices(snippet_length, int(data_gen_config['examples_no_snippets']), max_idx)
            for start, end in indices:
                X = group_sort.iloc[start : end]
                Y = group_sort.iloc[end+1 : end+1+label_length]
                X_a = np.array(X[search_val])
                Y_a = np.array(Y[search_val])
                snippets.append(Snippet(X_a, Y_a))

    except Exception as Argument:
        logging.error("converting dataset into snippets failed with message:")
        logging.error(str(Argument))

    return snippets



def make_interval_indices(length, no_intervals, max_idx):
    start_idxs = [random.randint(0, max_idx) for _ in range(no_intervals)]
    return [[x, x+length] for x in start_idxs]
