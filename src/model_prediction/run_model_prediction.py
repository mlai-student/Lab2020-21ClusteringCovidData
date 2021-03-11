import os, json, pickle, logging
import pandas as pd
import src.model_prediction.forecast_evaluation_functions as forecast_evaluation_functions
import src.model_prediction.benchmark_algorithms as forecast_functions
from src.data_representation.Examples import load_Examples_from_file
from src.data_representation.config_to_dict import get_config_dict


def forecast_a_snippet_list(forecasting_function, snippet_list):
    for snippet in snippet_list:
        snippet.forecast = forecasting_function(snippet.time_series)


def update_forecast_overview_file(
    main_config, forecast_dataset_filename, forecasting_function_name,
    forecast_evaluation_function_name, forecast_evaluation, data_foldername):
    # write get row from config
    cfg_settings_dict = get_config_dict(main_config)
    # add problem specific information.
    # append the prediciton to forecast.csv
    cfg_settings_dict["forecast_dataset_filename"] = forecast_dataset_filename
    cfg_settings_dict["forecast_function"] = forecasting_function_name
    cfg_settings_dict["forecast_evaluation_function"] = forecast_evaluation_function_name
    cfg_settings_dict["forecast_evaluation"] = forecast_evaluation
    csv_filename = data_foldername + "forecasting_results.csv"

    if os.path.isfile(csv_filename):
        df = pd.read_csv(csv_filename)
        pd_dict = pd.Series(cfg_settings_dict).to_frame().T
        model_df = df.append(pd_dict.iloc[0])
    else:
        model_df = pd.Series(cfg_settings_dict).to_frame().T
    model_df.to_csv(csv_filename, index=False)

def run_and_save_forecast_evaluation(main_config, dataset_examples, data_filename, data_foldername, forecasting_function_name):
    forecast_evaluation_function_name = main_config["model_prediction_settings"]["forecast_evaluation_function"]
    # Changed datset_examples.train_data to test_data
    forecast_evaluation = getattr(forecast_evaluation_functions, forecast_evaluation_function_name)(
        dataset_examples.test_data)
    forecast_dataset_filename = data_filename + "_w_forecast_" + forecast_evaluation_function_name + "_forecasting_fct_" + forecasting_function_name
    dataset_examples.save_to_file(forecast_dataset_filename)
    update_forecast_overview_file(
        main_config, forecast_dataset_filename, forecasting_function_name,
        forecast_evaluation_function_name, forecast_evaluation, data_foldername)


def forecast_example_set(main_config):
    # get dataset -> Examples to forecast on (maybe also a trained model depending on forecasting method)
    # Supposing that path is saved under "generated_data_path" in main_config
    data_filename = main_config["data_generating_settings"]["generated_data_path"]
    data_foldername = main_config["data_generating_settings"]["generated_folder_path"]
    model_filename = main_config["data_generating_settings"]["generated_model_path"]
    # load generatet dataset
    dataset_examples = load_Examples_from_file(data_filename)
    # run specified forecast, store the results as Examples file and compute precision of forecast
    forecasting_function_name = main_config["model_prediction_settings"]["forecast_function"]
    forecasting_function = getattr(forecast_functions, forecasting_function_name)

    #cluster forecasting methods
    if forecasting_function_name in ["lstm_forecast_cluster", "cluster_avg_forecast", "cluster_naive_forecast",
                                     "cluster_seasonal_naive_forecast", "linear_regression_cluster"]:
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
        forecasting_function(model, dataset_examples)
    #non cluster forecasting methods
    elif forecasting_function_name in ["lstm_forecast", "linear_regression"]:
        forecasting_function(dataset_examples)
    else:

        forecast_a_snippet_list(forecasting_function, dataset_examples.test_data)
    #now in dataset_examples in each snippet the "forecast" value is set.
    run_and_save_forecast_evaluation(main_config, dataset_examples, data_filename, data_foldername, forecasting_function_name)

def run_model_prediction_main(main_config):
    logging.debug("model_prediction.Run_model_prediction started main")
    forecast_example_set(main_config)
    logging.debug("model_prediction.Run_model_prediction ended main")
