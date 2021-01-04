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
from src.data_representation.Examples import load_Examples_from_file

# %%
import os
import pandas as pd
PROJECT_PATH = os.getcwd().replace("notebooks", "")
DATA_GEN_FOLDER_NAME = "Jan-04-2021"
DATASET_PATH = PROJECT_PATH + "data/" + DATA_GEN_FOLDER_NAME + "/"
OVERVIEW_DATASET_PATH = DATASET_PATH + "models.csv"
model_df = pd.read_csv(OVERVIEW_DATASET_PATH)

model_df.head()

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
