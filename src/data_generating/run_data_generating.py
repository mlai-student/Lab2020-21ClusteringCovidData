import pandas as pd
import logging

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
