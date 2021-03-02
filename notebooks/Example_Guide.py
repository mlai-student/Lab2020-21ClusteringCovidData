# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:nomarker
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Example Guide
#
# This notebook serves as an example, guiding you through some functionalities of the framework.\
# Visualization is both done with matplotlib and plotly dash.

# Adapting the path such that code from the src folder can be imported
import os
PROJECT_PATH = os.getcwd().replace("notebooks", "")
import sys
sys.path.append(PROJECT_PATH)
import src.model_training.clusters

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

# If you have not run the project yet, please do so by following the **Getting Started** section or using `python run.py --path_to_cfg config.ini`
#
# For the next step you should have at least one example file in a data folder. Please change the foldername corresponding to the date when you have run the script.

DATA_GEN_FOLDER_NAME = "Mar-02-2021"
DATASET_PATH = PROJECT_PATH + "data/" + DATA_GEN_FOLDER_NAME + "/"
OVERVIEW_DATASET_PATH = DATASET_PATH + "overview.csv"
FORECAST_DATASET_PATH = DATASET_PATH + "forecasting_results.csv"

# ### Overview file
#
# We first look at the overview file that was created during a run through the framework

import pandas as pd

model_df = pd.read_csv(OVERVIEW_DATASET_PATH)
model_df.head(5)

# As you can see, we have several columns describing, amongst other things, the settings of the config file, which includes the generated data path as well as the generated model path.
#
# We first show a simple visualization of one example using t-SNE

# TODO: TSNE

# # Cluster
# ### Visualizing 
# Each subplot corresponds to a cluster. The number above each plot states how many time-series snippets fall into this specific cluster

import pickle

example_no = 0
filename = model_df['data_generating_settings generated_model_path'][example_no]
with open(PROJECT_PATH + filename, 'rb') as f:
        model = pickle.load(f)
plt = model.plot_cluster()
plt.tight_layout()
plt.show()

ex = load_Examples_from_file(PROJECT_PATH + model_df['data_generating_settings generated_data_path'][example_no])
ratio = ex.test_data.shape[0]
pred_label = model.predict(ex, sample_weight=[i/ratio for i in range(1, ratio+1)])
plt.rcParams["figure.figsize"] = (10,2)
for label, sn in zip(pred_label, ex.test_data[:10]):
    print("predicted label: ", label)
    plt.plot(sn.time_series)
    plt.show()

# Next, we can also look at how the clusters are distributed onto the world map.\
# **Note**: This only works if you have chosen to cluster complete time-series.

if model_df['data_generating_settings complete_cluster'][example_no] == 'yes':
    model.plot_geo_cluster().show()
else:
    print("this only makes sense for complete time-series clustering")

# ### Scoring
#
# Sometime it makes sense to compare different scoring for the clustering methods. The framework provides a Silhouette Coefficient, the Calinski and Harabasz score, Davies-Bouldin score and a variance over the distribution of snippets into the cluster.

silhouette = model.silhouette()
calinski = model.calinski()
davies_bouldin = model.davies()
var, max_n_cluster, min_n_cluster = model.statistics()

print("The model: {}, achieved\nSilhouette: {:.2f}\nCalinski and Harabasz: {:.2f}\nDavies-Bouldin:{:.2f}".format\
      (model_df['model_training_settings models'][example_no], silhouette, calinski, davies_bouldin))

# # Forecasting
# If we choose to forecast during a run through, we get a forecasting results file. For us, the particular interesting part is in the last several columns. Here we get information about what forecaster was used and how it was evaluated. The result is in the last column.

forecast_df = pd.read_csv(FORECAST_DATASET_PATH)
forecast_df.head(5)

EXAMPLES_DATASET_PATH = PROJECT_PATH + forecast_df['forecast_dataset_filename'][example_no]

# ### Visualizing
#
# For an easier analysis of the results, we use box plots to visualize

from src.data_representation.Examples import load_Examples_from_file

examples = load_Examples_from_file(EXAMPLES_DATASET_PATH)
snippets = examples.train_data + examples.test_data
all_perc_list, hundret_perc_list, thousend_perc_list = [], [], []
samples_count = 0

for snippet in snippets:
    inverted_label = round(snippet.invert_to_abs_cases(snippet.label))
    inverted_forecast = round(snippet.invert_to_abs_cases(snippet.forecast))
    if inverted_label != 0:
        perc_dist = abs(inverted_forecast-inverted_label)/inverted_label
    else:
        perc_dist = 0
    perc_list.append(perc_dist)
    if inverted_label >= 100:
        hundret_perc_list.append(perc_dist)
    if inverted_label >= 1000:
        thousend_perc_list.append(perc_dist)
    samples_count += 1


all_df = pd.DataFrame(perc_list, columns=["All Snippets"])
h_df = pd.DataFrame(hundret_perc_list, columns=["Label >= 100"])
t_df = pd.DataFrame(thousend_perc_list, columns=["Label >= 1000"])

all_df.boxplot(vert=False, figsize=(13,5))
plt.show()
h_df.boxplot(vert=False, figsize=(13,5))
plt.show()
t_df.boxplot(vert=False, figsize=(13,5))
plt.show()
