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

overview_csv = pd.read_csv("../data/Mar-01-2021/overview.csv")
#print(overview_csv.columns)
examples = []
for idx, row in overview_csv.iterrows():
    example = load_Examples_from_file("../"+row["filename_example"])
    examples.append(example)
#overview_csv["data_generating_settings nr_days_for_avg"]

# %% [markdown]
# # TS visualization

# %%
#get one country out of every row 
country = []
for d in examples:
    country.append(list(filter(lambda snippet: snippet.country_id=="DEU", d.train_data))[0])


# %% [markdown]
# ## Smoothing Visualization

# %%
def plot_smoothing(with_scaler=True):
    plt.figure(figsize=(10,7))
    smoothing_idxs = [2,8,10]
    only_smoothing = [country[index] for index in smoothing_idxs]
    for idx, g in enumerate(only_smoothing):
        label = str(overview_csv.iloc[smoothing_idxs[idx]]["data_generating_settings nr_days_for_avg"])
        mult = g.scaler if with_scaler else 1  
        plt.plot(g.time_series*mult, label=label)
    plt.xlabel('Nr of days')
    
    if with_scaler:
        plt.title("Smoothed german timeseries with different nr_days_for_avg")
        plt.ylabel("Nr absolut daily cases")
    else:
        plt.title("Smoothed and Standardized german timeseries with different nr_days_for_avg")
        plt.ylabel("Nr standardized daily cases")
    plt.legend(loc="upper left")
    plt.show()

#With scalar -> before standardize
plot_smoothing(with_scaler=True)

#Without scalar -> after standardize -> real output
plot_smoothing(with_scaler=False)

# %% [markdown]
# ## TODO Augmentation visual

# %% [markdown]
# ## Snippet Visualizaition -> nessecary ?

# %%
