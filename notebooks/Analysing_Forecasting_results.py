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

#foldernames = ["linear_complete", "linear_cluster", "lstm_cluster", "lstm_all", "cluster_benchmark","non_cluster_benchmark"]
foldernames = ["naive_boxplot_variance_test"]
forecast_results_df= pd.DataFrame()
for foldername in foldernames:
    forecast_results_df= pd.concat([forecast_results_df,pd.read_csv(f"../data/{foldername}/forecasting_results.csv")])
    

# %%
non_group_cols = ['forecast_dataset_filename',
                  'data_generating_settings generated_folder_name',
                  'data_generating_settings generated_folder_path',
                  'data_generating_settings generated_data_path',
                  'data_generating_settings generated_model_path', 
                  'forecast_evaluation']
group_cols = [col for col in forecast_results_df.columns if not col in non_group_cols]

grouped_forecast_results = forecast_results_df.groupby(group_cols)
#group_cols

# %%
#Average results for each combination and save avg result in df with important cols:
avg_perc_list = ['0_avg_forecast_evaluation','100_avg_forecast_evaluation','1000_avg_forecast_evaluation']
avg_df = pd.DataFrame([], columns=group_cols+avg_perc_list)
for group in tqdm(grouped_forecast_results):
    forecast_filenames =  group[1]['forecast_dataset_filename'].values
    z_results, h_results, t_results = [],[],[]
    for filename in forecast_filenames:
        test_data = load_Examples_from_file(PROJECT_PATH+filename).test_data 
        z_results.append(avg_perc_dist(test_data,0))
        h_results.append(avg_perc_dist(test_data,100))
        t_results.append(avg_perc_dist(test_data,1000))
    means = [np.mean(z_results),np.mean(h_results),np.mean(t_results)]
    avg_df.loc[len(avg_df)] = (list(group[0]) + means)


# %%
#analyse after
results = pd.DataFrame()
for group in avg_df.groupby("model_prediction_settings forecast_function"):
    forecast_mean = group[1][avg_perc_list].mean()
    #pd.DataFrame(values, columns=["Label >= 100"]).boxplot(vert=False, figsize=(13,5))
    #plt.show()
    results[group[0]] = forecast_mean
results.T

# %% [markdown]
# # who benefits from smoothing:

# %%
#over all
print("Naive forecast results")
sm_results = pd.DataFrame()
for group in avg_df.groupby("data_generating_settings nr_days_for_avg"):
    naive_grp = group[1][group[1]["model_prediction_settings forecast_function"] == "naive_forecast"]
    forecast_mean = naive_grp[avg_perc_list].mean()
    #pd.DataFrame(values, columns=["Label >= 100"]).boxplot(vert=False, figsize=(13,5))
    #plt.show()
    sm_results[group[0]] = forecast_mean
sm_results.T

# %% [markdown]
# # overall winner

# %%
win_idxs = avg_df[avg_perc_list].idxmin()
print(win_idxs)
avg_df.loc[win_idxs.values].T

# %% [markdown]
# # variance test 

# %%

forecast_filenames =  forecast_results_df['forecast_dataset_filename'].values
z_results, h_results, t_results = [],[],[]
for filename in tqdm(forecast_filenames):
    test_data = load_Examples_from_file(PROJECT_PATH+filename).test_data 
    z_results.append(avg_perc_dist(test_data,0))
    h_results.append(avg_perc_dist(test_data,100))
    t_results.append(avg_perc_dist(test_data,1000))

results  = pd.DataFrame()
results["Label > 0"] = z_results
results["Label > 100"] = h_results
results["Label > 1000"] = t_results
results.boxplot(vert=False, figsize=(20,5))
plt.imgsave("naive_150_results_boxplot.png")
