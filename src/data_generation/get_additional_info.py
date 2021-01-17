#get additional info

import json

from src.data_generation.import_data import get_ecdc_dataset

#get additional snippet info as described in config
def get_additional_info(country_code, data_gen_config, data):
    categories = json.loads(data_gen_config["add_additional_info"])
    output = {}
    #add all the wanted categories
    for cat in categories:
        if cat[0] == "Population":
            output["Population"] = get_pop_data(country_code, cat, data_gen_config, data)
    return output


def get_pop_data(country_code, cat, data_gen_config, data):
    #load file and find country code
    #TODO change if other source should be used
    #TODO taka care about nan
    #df = get_ecdc_dataset(data_gen_config)
    #pop = (df[df["countriesAndTerritories"]==country_code]["popData2019"].iloc[0])
    pop = data["popData2019"].array[0]
    return (int) (pop)


def pop_dist_fct(x, y):
    return abs(x-y)


def get_additional_information_distance_functions(data_gen_config):
    categories = json.loads(data_gen_config["add_additional_info"])
    output = {}
    #add all the wanted categories
    for cat in categories:
        if cat[0] == "Population":
            output["Population"] = pop_dist_fct
    return output
