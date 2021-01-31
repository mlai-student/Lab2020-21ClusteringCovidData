import pandas as pd
import numpy as np
import logging
import random
from datetime import date
from pathlib import Path
from tqdm import tqdm
from src.data_representation.Snippet import Snippet
from src.data_representation.Examples import Examples
from src.data_generation.smoothing import smooth_timeline
from src.data_generation.augmentation import  data_augmentation
from src.data_generation.import_data import get_ecdc_dataset

from src.data_generation.get_additional_info import get_additional_info


# start the data generating process with a configuration set given
def run_data_generating_main(data_gen_config, filename):
    logging.debug("data_generating.Run_data_generating started main")
    # first get the raw data from ecdc csv using the url from config file
    df = get_ecdc_dataset(data_gen_config)

    snippet_examples = Examples()


    # TODO replace/fill NaN and resize cases but only after analysis beforehand -> control by config ini
    df = df[~(df["popData2019"].isnull())]
    if data_gen_config.getboolean("divide_by_country_population"):
        df["cases"] = df["cases"] / df["popData2019"] * 100
        #TODO!!!! das funktioniert so nicht
        #snippet_examples.invert_label_to_nr_cases.insert(0, invert)
    df.fillna(0, inplace=True)
    save_data_frame(df)


    if data_gen_config.getboolean("complete_cluster"):
        total_snippets = make_total_ts(df, data_gen_config)
        snippet_examples.fill_from_snippets(total_snippets, test_share=0., data_gen_config=data_gen_config)
        snippet_examples.add_padding()
    else:
        snippets = divide_ecdc_data_into_snippets(df, data_gen_config)
        if data_gen_config.getboolean("do_data_augmentation"):
            data_augmentation(snippets, data_gen_config)
        snippet_examples.fill_from_snippets(snippets, test_share=0.3, data_gen_config=data_gen_config)

    snippet_examples.save_to_file(filename)
    logging.debug("data_generating.Run_data_generating finished main")

def save_data_frame(df):
    try:
        today = date.today().strftime("%b-%d-%Y")
        Path("data/" + today).mkdir(parents=True, exist_ok=True)
        df.to_pickle("data/{}/{}".format(today, "ecdc_df"))
    except Exception as Argument:
        logging.error("Saving dataframe failed with following message:")
        logging.error(str(Argument))

def make_total_ts(ecdc_df, data_gen_config):
    examples = []
    try:
        country_group = ecdc_df.groupby(['countriesAndTerritories'], as_index=False)
        for country in country_group:
            ts = np.flipud(np.array(country[1]['cases'].array))
            invert_functions = []
            if data_gen_config.getboolean("replace_negative_values_w_zero"):
                ts[ts <0] =0
            country_code = country[1]['countryterritoryCode'].array[0]
            country_name = country[1]['countriesAndTerritories'].array[0]
            continent = country[1]['continentExp'].array[0]
            #if smoothing is wanted every value gets replaced by the nr_days_for_avg mean
            if data_gen_config.getboolean("do_smoothing"):
                output = smooth_timeline(ts, [], pd.Series(ts), 0, ts.shape[0], data_gen_config, invert_functions,use_zero_filler=True, no_Y=True)
                if type(output) == None:
                    continue
                else:
                    ts = output

            additional_info = get_additional_info(country[0], data_gen_config, country[1])

            examples.append(Snippet(ts, None, country_id=country_code, country=country_name,
                                    continent=continent, flip_order=False, additional_info = additional_info, invert_label_to_nr_cases=invert_functions))
    except Exception as Argument:
        logging.error("Storing time series failed with following message:")
        logging.error(str(Argument))
    return examples


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
        groups = ecdc_df.sort_values(by='dateRep', ascending=True).groupby(group_by)
        for group in groups:
            group_sort = group[1].sort_values(by='dateRep', ascending=True)
            max_idx = group_sort.shape[0] - snippet_length - label_length - 1
            indices = make_interval_indices(snippet_length, int(data_gen_config['examples_no_snippets']), max_idx)
            country_code = group[1]['countryterritoryCode'].array[0]
            country_name = group[1]['countriesAndTerritories'].array[0]
            continent = group[1]['continentExp'].array[0]
            for start, end in indices:
                invert_functions = []
                X = group_sort.iloc[start: end]
                Y = group_sort.iloc[end + 1: end + 1 + label_length]
                X_a, Y_a = np.array(X[search_val]), np.array(Y[search_val])
                if data_gen_config.getboolean("replace_negative_values_w_zero"):
                    X_a[X_a<0],Y_a[Y_a<0] = 0, 0
                #if smoothing is wanted every value gets replaced by the nr_days_for_avg mean
                if data_gen_config.getboolean("do_smoothing"):
                    output = smooth_timeline(X_a, Y_a, group_sort[search_val], start, end, data_gen_config,invert_functions)
                    if output == None:
                        continue
                    else:
                        X_a, Y_a = output
                additional_info = get_additional_info(group[0], data_gen_config, group[1])
                snippets.append(Snippet(X_a, Y_a, country_id=country_code, country=country_name,
                                        continent=continent, flip_order=False, additional_info = additional_info, invert_label_to_nr_cases=invert_functions))
    except Exception as Argument:
        logging.error("converting dataset into snippets failed with message:")
        logging.error(str(Argument))

    return snippets


def make_interval_indices(length, no_intervals, max_idx):
    start_idxs = [random.randint(0, max_idx) for _ in range(no_intervals)]
    return [[x, x + length] for x in start_idxs]
