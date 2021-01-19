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
from statistics import median
from tqdm import tqdm
# %matplotlib inline

from sklearn.metrics.pairwise import euclidean_distances as eukl_dist

# %%
import os
import pandas as pd
PROJECT_PATH = os.getcwd().replace("notebooks", "")
DATA_GEN_FOLDER_NAME = "Jan-19-2021"
DATASET_PATH = PROJECT_PATH + "data/" + DATA_GEN_FOLDER_NAME + "/"
OVERVIEW_DATASET_PATH = DATASET_PATH + "overview.csv"
overview_df = pd.read_csv(OVERVIEW_DATASET_PATH)

overview_df.head(12)

# %%
#distance to zero
for idx, row in overview_df.iterrows():
    with open(PROJECT_PATH+ row["filename"], 'rb') as f:
        dataset = pickle.load(f)
    eukl_distances = [(ts.country,eukl_dist([ts.to_vector()], [np.zeros(len(ts.to_vector()))])) for ts in dataset.train_data]
    eukl_distances = sorted(eukl_distances, key=lambda x: x[1], reverse=True)
    eukl_sizes = [entry[1] for entry in eukl_distances]
    print(eukl_distances[:5])
    start_val = 0
    ax = plt.gca()
    #ax.set_yscale('log')
    print(f"Median= {median(eukl_sizes[start_val:])}")
    plt.scatter(list(range(len(eukl_sizes[start_val:]))),eukl_sizes[start_val:])
    plt.show()

# %%
#distance between all pairs colored 
#distance to zero
for idx, row in tqdm(overview_df.iterrows()):
    with open(PROJECT_PATH+ row["filename"], 'rb') as f:
        dataset = pickle.load(f)
    for x_ts in dataset.train_data:
        eukl_distances = [eukl_dist([ts.to_vector()], [x_ts.to_vector()]) for ts in dataset.train_data]
        eukl_sizes = sorted(eukl_distances, reverse=True)
        print(len(eukl_sizes))
        plt.scatter(list(range(len(eukl_sizes))),eukl_sizes)
    plt.show()
