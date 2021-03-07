# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Packages and data input

# %%
import os,sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

PROJECT_PATH = os.getcwd().replace("notebooks", "")
sys.path.append(PROJECT_PATH)
from src.data_representation.Examples import load_Examples_from_file
from src.model_prediction.forecast_evaluation_functions import avg_perc_dist

foldername = "linear_complete"

forecast_results_df = pd.read_csv(f"../data/{foldername}/forecasting_results.csv")

forecast_results_df

# %%
non_group_cols = ['forecast_dataset_filename',
                  'data_generating_settings generated_folder_name',
                  'data_generating_settings generated_folder_path',
                  'data_generating_settings generated_data_path',
                  'data_generating_settings generated_model_path', 
                  'forecast_evaluation']
group_cols = [col for col in forecast_results_df.columns if not col in non_group_cols]

grouped_forecast_results = forecast_results_df.groupby(group_cols)
group_cols

# %%
#Average results for each combination and save avg result in df with important cols:
avg_df = pd.DataFrame([], columns=group_cols+['avg_forecast_evaluation'])
for group in (grouped_forecast_results):
    forecast_filenames =  group[1]['forecast_dataset_filename'].values
    results = []
    for filename in forecast_filenames:
        results.append(avg_perc_dist(load_Examples_from_file(PROJECT_PATH+filename).test_data,100))
    avg_df.loc[len(avg_df)] = (list(group[0]) + [np.mean(results)])
    print(results)


# %%
#analyse after
for group in avg_df.groupby("data_generating_settings nr_days_for_avg"):
    values = group[1]["avg_forecast_evaluation"].values
    forecast_mean = np.mean(values)
    #pd.DataFrame(values, columns=["Label >= 100"]).boxplot(vert=False, figsize=(13,5))
    #plt.show()
    print(f"{group[0]}: {forecast_mean}")
