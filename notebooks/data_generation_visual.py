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
#PACKAGES
import os
PROJECT_PATH = os.getcwd().replace("notebooks", "")
import sys
sys.path.append(PROJECT_PATH)
from src.data_representation.Examples import load_Examples_from_file

# %%
#running the project from here wont work since we demand to be in the main directory
#Display the log file:

log_file =  open(PROJECT_PATH+"run.log", "r")
print(log_file.read())
log_file.close()





# %% [markdown]
# **Visualize data_generation output**

# %%
#name of the folder where the output of the Example class file is located
DATA_GEN_FOLDER_NAME = "Dec-09-2020"
EXAMPLES_DATASET_PATH = PROJECT_PATH + "data/" + DATA_GEN_FOLDER_NAME + "/total_snippets"

snippets = load_Examples_from_file(EXAMPLES_DATASET_PATH)

# %%
print("Number train examples: " + str(len(snippets.train_examples)))
print("Number test examples: " + str(len(snippets.test_examples)))
