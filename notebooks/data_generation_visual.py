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
# **Current log file**

# %%
# PACKAGES
import os
PROJECT_PATH = os.getcwd().replace("notebooks", "")
import sys
sys.path.append(PROJECT_PATH)
from src.data_representation.Examples import load_Examples_from_file
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import src.model_training.clusters
from tqdm import tqdm

# %%
# running the project from here wont work since we demand to be in the main directory
# Display the log file:

log_file = open(PROJECT_PATH + "run.log", "r")
print(log_file.read())
log_file.close()

# %% [markdown]
# **Visualize/Analyse data_generation output**

# %%
# name of the folder where the output of the Example class file is located
DATA_GEN_FOLDER_NAME = "Feb-09-2021"
EXAMPLES_DATASET_PATH = PROJECT_PATH + "data/" + DATA_GEN_FOLDER_NAME + "/0"

snippets = load_Examples_from_file(EXAMPLES_DATASET_PATH)

# %%
print("Number train examples: " + str(len(snippets.train_data)))
print("Number test examples: " + str(len(snippets.test_data)))

# %%
#Display all snippets in one plot to identify maybe classes of errors
all_snippets = snippets.train_data + snippets.test_data
for snippet in tqdm(all_snippets):
    if snippet.scaler > 50000:
        print(f"Country: {snippet.country}")
        print(f"Standardized label: {snippet.label}")
        print(f"Unstandardized label: {snippet.original_label[0]}")
        plt.plot(snippet.time_series)
        plt.show()
        plt.plot(snippet.time_series*snippet.scaler)
        
        break
plt.show()


# %%

# %%
#get an overview over the values in all time series using pd hist()
values = []
for snippet in all_snippets:
    for value in snippet.time_series:
        values.append(value)
pd.DataFrame(values).hist(bins=101)
#closer look at smaller values (<bound)
bound = 10000
pd.DataFrame([x for x in values if (x <bound and x!=0)]).hist(bins=101)



#nr/proportion of zero entries: 
zero_counter = len([i for i, x in enumerate(values) if x == 0])
print("Number zero entries: " + str(zero_counter) + " Proportion: " + str(zero_counter/len(values)))
print("Total number entries: " + str(len(values)))

# %%
#analyse the values itself a little bit: 
print("mean: "  + str(np.mean(values)))
print("varianz: " + str(np.var(values)))
ten_quantiles =[np.quantile(values, (i+1)*.10) for i in range(10)]
print("quantiles (10%, 20% ...): " +str(ten_quantiles))
four_quantiles =[np.quantile(values, (i+1)*.25) for i in range(4)]
print("four quantiles: " + str(four_quantiles))


#test if there are negative values:
neg_values = [x for  x in values if x<0]
print("Number of negative values: " + str(len(neg_values)))
print("Negative values: " + str(neg_values))

# %% [markdown]
# # Forecast evaluation visual

# %%
# name of the folder where the output of the Example class file is located
DATA_GEN_FOLDER_NAME = "Feb-09-2021"
EXAMPLES_DATASET_PATH = PROJECT_PATH + "data/" + DATA_GEN_FOLDER_NAME + "/0_w_forecast_avg_perc_dist_forecasting_fct_naive_forecast"

examples = load_Examples_from_file(EXAMPLES_DATASET_PATH)
snippets = examples.train_data+examples.test_data
all_perc_list=[]
hundret_perc_list = []
thousend_perc_list = []
samples_count = 0
for snippet in snippets:
    inverted_label = round(snippet.invert_to_abs_cases(snippet.label))
    
    if inverted_label < 0:
        print(f"Snippet label: {snippet.label} Snippet forecast {snippet.forecast} and inverted label {inverted_label}")
    if inverted_label != 0:
        inverted_forecast = round(snippet.invert_to_abs_cases(snippet.forecast))
        if inverted_forecast < 0:
            print(f"Snippet label: {snippet.label} Snippet forecast {snippet.forecast} and inverted forecast {inverted_forecast}")

        perc_dist = abs(inverted_forecast-inverted_label)/inverted_label
        perc_list.append(perc_dist)
        if inverted_label >= 100:
            hundret_perc_list.append(perc_dist)
        if inverted_label >= 1000:
            thousend_perc_list.append(perc_dist)
        samples_count += 1

all_df = pd.DataFrame(perc_list, columns=["All Snippets"])
h_df = pd.DataFrame(hundret_perc_list, columns=["Label >= 100"])
t_df = pd.DataFrame(thousend_perc_list, columns=["Label >= 1000"])
print(np.var(perc_list))
print(np.var(hundret_perc_list))
print(np.var(thousend_perc_list))
all_df.boxplot(vert=False, figsize=(13,5))
plt.show()
h_df.boxplot(vert=False, figsize=(13,5))
plt.show()
t_df.boxplot(vert=False, figsize=(13,5))
plt.show()
