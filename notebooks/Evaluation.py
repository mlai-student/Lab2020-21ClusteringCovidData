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
# %matplotlib inline

# %%
import os
import pandas as pd
PROJECT_PATH = os.getcwd().replace("notebooks", "")
DATA_GEN_FOLDER_NAME = "Jan-04-2021"
DATASET_PATH = PROJECT_PATH + "data/" + DATA_GEN_FOLDER_NAME + "/"
OVERVIEW_DATASET_PATH = DATASET_PATH + "models.csv"
model_df = pd.read_csv(OVERVIEW_DATASET_PATH)

model_df.shape

# %%
import copy
import pickle
from collections import OrderedDict

score_overview = copy.deepcopy(model_df)
sil, cal, dav = [], [], []
for ind in score_overview.index:
    filename = score_overview['filename'][ind]
    with open(DATASET_PATH + "model/" + filename, 'rb') as f:
        model = pickle.load(f)
        sil.append(model.silhouette())
        cal.append(model.calinski())
        dav.append(model.davies())
        
keys = ["silhouette_score", "Calinski_harabasz_index", "davies_bouldin_index"]  
values = [sil, cal, dav]
score_df = pd.DataFrame(OrderedDict(zip(keys, values)))
score_overview = pd.concat([score_overview, score_df], axis=1)

score_overview

# %% [markdown]
# # Visualization

# %% [markdown]
# ## Bar plot depending on arbitrary item

# %%
#one of value_overview.columns 
#or special case clustering method
cols = score_overview.columns
attr_of_interest = "model_name"
evaluation_cols = cols[8:len(cols)]
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
evaluation_cols = cols[8:len(cols)]
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
first_attr_of_interest = "model_name"
sec_attr_of_interest = "no_cluster"
evaluation_cols = cols[8:len(cols)]
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
