import pandas as pd
import numpy as np
import random, logging
from tqdm import tqdm
from copy import deepcopy as dc
import datetime
from src.data_representation.Snippet import Snippet
from src.data_representation.Examples import Examples
from src.data_generation.smoothing import smooth_timeline
from src.data_generation.augmentation import  data_augmentation
from src.data_generation.import_data import get_ecdc_dataset
from src.data_generation.get_additional_info import get_additional_info

#add missing values inside a TimeSeries
def add_missing_values_in_between(df):
    co_grp = df.groupby("countriesAndTerritories")
    new_rows = []
    for country in tqdm(co_grp):
        min_date, max_date = country[1]["dateRep"].min(), country[1]["dateRep"].max()
        tmp_date = min_date
        while (max_date-tmp_date).days != 0:
            if not tmp_date in list(country[1]["dateRep"].values):
                #add a zero entry
                new_row = dc(country[1].iloc[0])
                new_row["dateRep"] = tmp_date
                new_row["day"], new_row["month"], new_row["year"] = tmp_date.day, tmp_date.month, tmp_date.year
                new_row["cases"], new_row["deaths"] = 0,0
                new_rows.append(new_row)
            tmp_date +=datetime.timedelta(days=1)
    new_ecdc_df = pd.DataFrame(new_rows, columns=df.columns)
    df = pd.concat([df, new_ecdc_df])
    #sort the new entries in the correct order
    df = df.sort_values(by ='dateRep')


# start the data generating process with a configuration set given
def run_data_generating_main(data_gen_config, fix_cfg, filename):
    logging.debug("data_generating.Run_data_generating started main")
    # first get the raw data from ecdc csv using the url from config file
    df = get_ecdc_dataset(fix_cfg)
    #get empty exmples object to fill with snippets or complete country time series
    snippet_examples = Examples()

    if data_gen_config.getboolean("divide_by_country_population"):
        df = df[~(df["popData2019"].isnull())]
        df["cases"] = df["cases"] / df["popData2019"] * 100
    df.fillna(0, inplace=True)
    if fix_cfg["general_settings"].getboolean("replace_negative_values_w_zero"):
        df["cases"] = df["cases"].clip(lower=0)
    if fix_cfg["general_settings"].getboolean("add_missing_values_inside_ts"):
        add_missing_values_in_between(df)

    save_ecdc_data_frame(df, data_gen_config)

    #check whether snippets or complete clusters are wanted and fill the snippet_examples respectively
    if data_gen_config.getboolean("complete_cluster"):
        total_snippets = make_total_ts(df, data_gen_config)
        snippet_examples.fill_from_snippets(total_snippets, data_gen_config=data_gen_config)
        snippet_examples.add_padding()
    else:
        #get a random subset of countries for the test and train data:
        test_share = float(data_gen_config["test_country_share"])
        countries = df['countriesAndTerritories'].unique()
        test_countries = np.random.choice(countries, (int)(test_share*len(countries)), replace=False)
        test_data = df.loc[df['countriesAndTerritories'].isin(test_countries)]
        train_data = df.loc[~df['countriesAndTerritories'].isin(test_countries)]
        test_snippets = divide_into_snippets(test_data, data_gen_config)
        train_snippets = divide_into_snippets(train_data, data_gen_config)
        if data_gen_config.getboolean("do_data_augmentation"):
            data_augmentation(test_snippets, data_gen_config)
            data_augmentation(train_snippets, data_gen_config)
        snippet_examples.fill_from_snippets(train_snippets, test_snippets=test_snippets, data_gen_config=data_gen_config)
    snippet_examples.standardize()

    snippet_examples.save_to_file(filename)
    logging.debug("data_generating.Run_data_generating finished main")

def save_ecdc_data_frame(df, data_gen_cfg):
    try:
        df.to_pickle(data_gen_cfg["generated_folder_path"]+ "ecdc_df")
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

def divide_into_snippets(ecdc_df, data_gen_config):
    snippets = []
    search_val = "cases"
    group_by = "countriesAndTerritories"
    try:
        snippet_length = int(data_gen_config['examples_snippet_length'])
    except ValueError as Argument:
        logging.error("Converting config to string failed with message:")
        logging.error(str(Argument))
    try:
        # extract Examples from data
        groups = ecdc_df.sort_values(by='dateRep', ascending=True).groupby(group_by)
        for group in groups:
            group_sort = group[1].sort_values(by='dateRep', ascending=True)
            max_idx = group_sort.shape[0] - snippet_length -1
            indices = make_interval_indices(snippet_length, int(data_gen_config['examples_no_snippets']), max_idx)
            country_code = group[1]['countryterritoryCode'].array[0]
            country_name = group[1]['countriesAndTerritories'].array[0]
            continent = group[1]['continentExp'].array[0]
            for start, end in indices:
                invert_functions = []
                X, Y = group_sort.iloc[start: end], group_sort.iloc[end: end + 1]
                X_a, Y_a = np.array(X[search_val]), np.array(Y[search_val])
                Y_orig = dc(Y_a)
                #if smoothing is wanted every value gets replaced by the nr_days_for_avg mean
                if data_gen_config.getboolean("do_smoothing"):
                    output = smooth_timeline(X_a, Y_a, group_sort[search_val], start, end, data_gen_config,invert_functions)
                    if output == None:
                        continue
                    else:
                        X_a, Y_a = output
                else:
                    Y_a = Y_a[0]
                additional_info = get_additional_info(group[0], data_gen_config, group[1])
                snippets.append(Snippet(X_a, Y_a, original_label=Y_orig, country_id=country_code, country=country_name,
                                        continent=continent, flip_order=False, additional_info = additional_info, invert_label_to_nr_cases=invert_functions))
    except Exception as Argument:
        logging.error("converting dataset into snippets failed with message:")
        logging.error(str(Argument))
    return snippets

def make_interval_indices(length, no_intervals, max_idx):
    start_idxs = [random.randint(0, max_idx) for _ in range(no_intervals)]
    return [[x, x + length] for x in start_idxs]
