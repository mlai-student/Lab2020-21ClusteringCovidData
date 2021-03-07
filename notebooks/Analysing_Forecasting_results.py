# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
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

foldername = "linear_cluster"

forecast_results_df = pd.read_csv(f"../data/{foldername}/forecasting_results.csv")

forecast_results_df = forecast_results_df[forecast_results_df['model_training_settings models'] == 'KMeans']

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
        results.append(avg_perc_dist(load_Examples_from_file(PROJECT_PATH+filename).test_data))
    avg_df.loc[len(avg_df)] = (list(group[0]) + [np.mean(results)])
    # print(results)


# %%
avg_df

# %%
#analyse after
for group in avg_df.groupby(["data_generating_settings nr_days_for_avg", "model_training_settings n_clusters"]):
    values = group[1]["avg_forecast_evaluation"].values
    forecast_mean = np.mean(values)
    #pd.DataFrame(values, columns=["Label >= 100"]).boxplot(vert=False, figsize=(13,5))
    #plt.show()
    print(f"{group[0]}: {forecast_mean}")

# %% [markdown]
# # Results
#
# ## Linear 
# ### Cluster 100: 
# -1: 0.27008258552985753 \
# 3: 0.33088937448977473 \
# 7: 0.38978436335074584 
#
# ### 50:
# -1: 0.3011381067200981 \
# 3: 0.36559987340287414 \
# 7: 0.4316047958630479 
#
# ### Complete 100:
# -1: 0.2764243281868022 \
# 3: 0.3351384753593156\
# 7: 0.37557498530381717
#
# ### 50:
# -1: 0.30760563419867787\
# 3: 0.37275411780280554\
# 7: 0.4151329015080114
#
#
# ## LSTM
# ### complete 100:
# 0.32384209200842 \
# 3: 0.3707301242947537 \
# 7: 0.38836846132681435
# ### 50: 
# 0.35635699590258285\
# 3: 0.4059244121221388\
# 7: 0.42690407178808726
#
# ### 0:
# -1: 0.8193623707720838\
# 3: 0.8739256446515782\
# 7: 0.9442986509723494
# #### grouped by cluster
# 5: 0.9082094530654956\
# 10: 0.7951428182911551\
# 15: 0.867058002367925\
# 20: 0.9463719481367731
#
# ### cluster 100:
# -1: 0.30451311125828906 \
# 3: 0.38341639878129646 \
# 7: 0.42842988052459047
# ### 50:
# 0.3330582208779861\
# 3: 0.41909458590327253\
# 7: 0.47933431795026654
