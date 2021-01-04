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

# %% [markdown]
# # Showcase Notebook for Usage Examples

# %% [markdown]
# ### Import Project and Cluster Methods

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

# %% [markdown]
# ### Load Examples

# %%
import os
PROJECT_PATH = os.getcwd().replace("notebooks", "")
DATA_GEN_FOLDER_NAME = "Jan-03-2021"
DATASET_PATH = PROJECT_PATH + "data/" + DATA_GEN_FOLDER_NAME + "/"
OVERVIEW_DATASET_PATH = DATASET_PATH + "overview.csv"
overview_df = pd.read_csv(OVERVIEW_DATASET_PATH)

overview_df

# %%
#load files from filename col and apply models to that with different cluster sizes
data = []
for f in overview_df['filename']:
    data.append((load_Examples_from_file(PROJECT_PATH + str(f)), f))

    
from tqdm import tqdm
n_clusters = [3, 4, 5, 10]
models = [cl.KMedoids, cl.KMeans,  cl.DBSCAN, cl.TS_KMeans, cl.TS_KShape]
for n in n_clusters:
    filenames = []
    for m in tqdm(models):
        model_names = []
        for ex,file in data:
            ex.add_padding()
            model = m(n, metric='euclidean').fit(ex)
            filenr = file.split("/")[-1]
            filename = f"{model.name}_no_cluster_{n}_{filenr}"
            model.save_model(filename)
            model_names.append(filename)
        filenames.append(model_names)
    overview_df[f'kmedoids_{n}_filename'] = filenames[0]
    overview_df[f'kmeans_{n}_filename'] = filenames[1]
    overview_df[f'dbscan_{n}_filename'] = filenames[2]
    overview_df[f'ts_kmeans_{n}_filename'] = filenames[3]
    overview_df[f'ts_kshape_{n}_filename'] = filenames[4]
overview_df.to_csv(DATASET_PATH + "overview_filenames.csv")

# %%
import copy
import pickle
value_overview = copy.deepcopy(overview_df).to_numpy()
for col in range(7,value_overview.shape[1]):
    for row in range(value_overview.shape[0]):
        filename = value_overview[row][col]
        with open(DATASET_PATH + "model/" + filename, 'rb') as f:
            model = pickle.load(f)
            var, n_max, n_min = model.statistics()
            value_overview[row][col] = var

            
value_overview = pd.DataFrame(value_overview, columns=overview_df.columns)
value_overview

# %%
#old code used to find min -> TODO verbessern
comment = '''
df_ = pd.DataFrame(value_overview, columns=var_df.columns)
df_.min()
ts_k = df_[df_["ts_kshape_10_filename"] <= 171]
print(ts_k)
# print(overview_df.iloc[4])
with open(DATASET_PATH + "model/TS_KShape_no_cluster_10_46", 'rb') as f:
            ts_kshape_10 = pickle.load(f)
        
ts_kshape_10.plot_geo_cluster().show()
ts_kshape_10.plot_cluster().show()
'''

# %% [markdown]
# ### Demonstration of Augmentation/Smoothing

# %%
germany = []
for d in data:
    germany.append(list(filter(lambda snippet: snippet.country_id=="DEU", d[0].train_data))[0])

# %%
import matplotlib.pyplot as plt
plt.figure()
for g in germany:
    plt.plot(g.time_series)
plt.xlabel('Different methodologys on german time series')
plt.show()

# %% [markdown]
# # Visualizing var_df 
#

# %% [markdown]
# ## Bar plot depending on arbitrary item 

# %%
#one of value_overview.columns 
#or special case clustering method
attr_of_interest = "do_smoothing"
bar_df = []
bar_df_labels = []
for table in value_overview.groupby(attr_of_interest):
    avg_value = 0 
    for row in table[1].iterrows():
        avg_value+= row[1].iloc[7:table[1].shape[1]].sum()/(table[1].shape[1]-7)
    avg_value/=table[1].shape[0]
    bar_df_labels.append(str(table[0]))
    bar_df.append(avg_value)


pd.DataFrame([bar_df], columns=bar_df_labels).plot.bar()

# %% [markdown]
# ## 2 dim plot showing realtion between two 

# %%
import plotly.express as px
df = px.data.tips()

fig = px.density_heatmap(df, x="total_bill", y="tip")
fig.show()
