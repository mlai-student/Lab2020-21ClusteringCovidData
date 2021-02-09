# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import warnings
warnings.filterwarnings('ignore')

import os
PROJECT_PATH = os.getcwd().replace("notebooks", "")
import sys
sys.path.append(PROJECT_PATH)
import src.model_training.clusters as cl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_representation.Examples import load_Examples_from_file
import pickle
# %matplotlib inline

# %%
import os
import pandas as pd
PROJECT_PATH = os.getcwd().replace("notebooks", "")
DATA_GEN_FOLDER_NAME = "Feb-02-2021"
DATASET_PATH = PROJECT_PATH + "data/" + DATA_GEN_FOLDER_NAME + "/"
OVERVIEW_DATASET_PATH = DATASET_PATH + "overview.csv"
model_df = pd.read_csv(OVERVIEW_DATASET_PATH)

model_df.head(6)

# %%
FORECAST_DATASET_PATH = DATASET_PATH + "forecasting_results.csv"
forecast_df = pd.read_csv(FORECAST_DATASET_PATH)
forecast_df.head()

# %%
import copy
import pickle
from collections import OrderedDict
from tqdm import tqdm

score_overview = copy.deepcopy(model_df)
sil, cal, dav = [], [], []
sil_pop, cal_pop, dav_pop = [], [], []
var_l = []
i = 0
max_cluster = []
for ind in tqdm(score_overview.index):
    filename = score_overview['filename'][ind]
    with open(DATASET_PATH + "model/" + filename, 'rb') as f:
        model = pickle.load(f)
        sil.append(model.silhouette(metric =score_overview['metric'][ind]))
        cal.append(model.calinski())
        dav.append(model.davies())
        var, _, _ = model.statistics()
        var_l.append(var)
        max_cluster.append(max(model.n_per_clusters))
        sil_pop.append(model.silhouette(use_add_info=True, key="Population"))
        cal_pop.append(model.calinski(use_add_info=True, key="Population"))
        dav_pop.append(model.davies(use_add_info=True, key="Population"))
        
keys = ["silhouette", "Calinski", "davies", "var", "silhouette_pop", "Calinski_pop", "davies_pop", "max_cluster"]  
values = [sil, cal, dav, var_l, sil_pop, cal_pop, dav_pop, max_cluster]

# keys = ["silhouette", "Calinski", "davies", "var", "max_cluster"]  
# values = [sil, cal, dav, var_l, max_cluster]

score_df = pd.DataFrame(OrderedDict(zip(keys, values)))
score_overview = pd.concat([score_overview, score_df], axis=1)

score_overview.head()

# %%
model_df['data_generating_settings nr_days_for_avg']

# %%
import pickle
# cond_df = score_overview[(score_overview.no_cluster==7) & (score_overview["metric"]=="euclidean") & (score_overview.do_smoothing=="yes")]
# # (score_overview["metric"]=="dtw") & (score_overview.no_cluster==5) & 
# best_row = cond_df['silhouette'].idxmax()
best_row = 0
filename = model_df['main_flow_settings generated_model_path'][best_row]
with open(DATASET_PATH + "model/" + filename, 'rb') as f:
        model = pickle.load(f)
model.plot_cluster().show()

# %%
a = model.clusters[0].train_data[1].time_series
plt.rcParams['figure.figsize'] = [20, 5]
plt.plot(a)

# %%
model.plot_geo_cluster()

# %%
score_overview.drop(columns=["filename"]).head(10)

# %% [markdown]
# # Visualization

# %% [markdown]
# ## Bar plot depending on arbitrary item

# %%
#one of value_overview.columns 
#or special case clustering method
cols = score_overview.columns
attr_of_interest = "model_name"
evaluation_cols = cols[9:len(cols)]
bar_values = pd.DataFrame([], index=evaluation_cols)

for table in score_overview.groupby(attr_of_interest):
    bar_values[table[0]] = table[1][evaluation_cols].mean()

for index, row in bar_values.iterrows():
    print("Bar plot for: " + str(index))
    row.plot.bar()
    plt.show()

# %% [markdown]
# ## 2 dim plot showing realtion between two

# %%
first_attr_of_interest = "model_name"
sec_attr_of_interest = "nr_days_for_avg"
evaluation_cols = cols[9:len(cols)]
first_attr_evals = []
for first_attr_table in score_overview.groupby(first_attr_of_interest):
    first_attr_eval = pd.DataFrame([], index=evaluation_cols)
    for sec_attr_table in first_attr_table[1].groupby(sec_attr_of_interest):
        first_attr_eval[sec_attr_table[0]] = sec_attr_table[1][evaluation_cols].mean()
    first_attr_evals.append([first_attr_table[0], first_attr_eval])

#sort after eval cols
bar_values = [] 
for index, col in enumerate(evaluation_cols):
    eval_result = pd.DataFrame([], columns=first_attr_evals[0][1].columns)
    for  first_attr_eval in first_attr_evals:
        eval_result = eval_result.append(pd.Series(first_attr_eval[1].loc[col], name=first_attr_eval[0]))
    bar_values.append([col, eval_result])

  
for table in bar_values:
    print("Bar plot for: " + str(table[0]))
    plt.figure(figsize=(7,7))
    sns.heatmap(table[1], annot=True)
    plt.gca().set_ylim(len(table[1].index)+0.5, -0.5)
    plt.yticks(rotation=0)
    plt.show()

# %%
sns.relplot(y="silhouette", x="nr_days_for_avg", hue="divide_by_country_population",
            style="metric", kind="line", dashes=True, markers=True, data=score_overview);
plt.savefig('plot_dtw_eucl.png')

# %%
sns.relplot(y="silhouette_pop", x="nr_days_for_avg", hue="divide_by_country_population",
            style="metric", kind="line", dashes=True, markers=True, data=score_overview);

# %%
first_attr_of_interest = "model_name"
sec_attr_of_interest = "no_cluster"
evaluation_cols = cols[9:len(cols)]
first_attr_evals = []
for first_attr_table in score_overview.groupby(first_attr_of_interest):
    first_attr_eval = pd.DataFrame([], index=evaluation_cols)
    for sec_attr_table in first_attr_table[1].groupby(sec_attr_of_interest):
        first_attr_eval[sec_attr_table[0]] = sec_attr_table[1][evaluation_cols].mean()
    first_attr_evals.append([first_attr_table[0], first_attr_eval])

#sort after eval cols
bar_values = [] 
for index, col in enumerate(evaluation_cols):
    eval_result = pd.DataFrame([], columns=first_attr_evals[0][1].columns)
    for  first_attr_eval in first_attr_evals:
        eval_result = eval_result.append(pd.Series(first_attr_eval[1].loc[col], name=first_attr_eval[0]))
    bar_values.append([col, eval_result])

  
for table in bar_values:
    print("Bar plot for: " + str(table[0]))
    plt.figure(figsize=(7,7))
    sns.heatmap(table[1], annot=True)
    plt.gca().set_ylim(len(table[1].index)+0.5, -0.5)
    plt.yticks(rotation=0)
    plt.show()
