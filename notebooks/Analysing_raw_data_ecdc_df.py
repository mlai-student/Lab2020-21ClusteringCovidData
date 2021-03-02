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
# # Packages and paths and read data

# %%
import os,sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from copy import deepcopy as dc

PROJECT_PATH = os.getcwd().replace("notebooks", "")
sys.path.append(PROJECT_PATH)
from src.data_generation.import_data import get_standard_ecdc_dataset

ecdc_df = get_standard_ecdc_dataset()

# %% [markdown]
# # Data Overview

# %%
#general overview over the given dataset:

print(f"The dataset contains: {ecdc_df.shape[0]} number of entries")

cols = ecdc_df.columns

print(f"In total there are the following number of unique values per attribute:")
print(ecdc_df.nunique())

cases_sum = ecdc_df["cases"].sum()
print(f"\n In total there are {cases_sum} COVID cases counted in the dataset")


#timespan
min_date, max_date = ecdc_df["dateRep"].min(), ecdc_df["dateRep"].max()
def get_day(date):
    return f"{date.month}/{date.day}/{date.year}"
print(f"\n Dates range from {get_day(min_date)} to {get_day(max_date)}")
print(f"So in total {(max_date-min_date).days} days")

# %% [markdown]
# ## Value Distribution

# %%
#Cases Value distributions:
#ecdc_df.isnull().sum()
print("There are no NAN values in the dataset")
nr_neg_cases = ecdc_df[ecdc_df["cases"]<0].shape[0]
print(f"But there are {nr_neg_cases} negative values in the cases attribute Those are reaplaces by 0")
ecdc_df["cases"] = ecdc_df["cases"].clip(lower=0)


print(f"The cases Values are distributed in the following way:")
nr_zeros = ecdc_df[ecdc_df["cases"]==0].shape[0]
print(f"There are {nr_zeros} zero entries in the cases attribute. So {round(nr_zeros/ecdc_df.shape[0]*100,3)}% of all entries are zero.")


#CDF
# Get the frequency, PDF and CDF for each value in the series
df_series = ecdc_df[ecdc_df["cases"]!=0]["cases"]
df = pd.DataFrame(df_series)
# Frequency
stats_df = df.groupby('cases') ['cases'] \
.agg('count') \
.pipe(pd.DataFrame) \
.rename(columns = {'cases': 'frequency'})

# PDF
stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])

# CDF
stats_df['cdf'] = stats_df['pdf'].cumsum()
stats_df = stats_df.reset_index()

print(f"\n In the following one can see the cummulative distribution over the non zero cases attribute values.")
print("Here one can see that about 50% percent of all reported non zero cases cases are below 100.")
print("In the Results section of the report we have shown that the prediction performance increases significantly when ignoring such small case reports")
stats_df.plot(x = 'cases', y = ['cdf'], grid = True)
plt.xscale("log")
plt.ylabel("Probability")
plt.title("Cummulative Distribution Function of the cases attribute")
plt.show()


# %%
#Nr reports per country distribution
#missing country population data
print(f"Not all countries gave the same amount of data points:")
country_group = ecdc_df.groupby("countriesAndTerritories")
country_nr_result = {}
for country in country_group:
    country_nr_result[country[0]] = country[1].shape[0]
country_nr_df = pd.DataFrame.from_dict(country_nr_result, orient="index")
country_nr_df.hist(bins=15)
plt.xlabel("Nr datapoints per country in bins")
plt.ylabel("Nr countries per bin")
plt.title("Distribution of Datapoints per country")
plt.show()

# %%
#???missing days in between 
#-> wenn wir das reporten sollte wir es auch einbauen!

print(f"Most of the time the reason behind less data points is that the country started reporting dayly cases after {get_day(min_date)}")
print("But it also occurs that there are days missing in between:")

def check_in_between_missing_values(df):
    co_grp = df.groupby("countriesAndTerritories")
    total_nr_days_missing = 0
    nr_countr = 0
    for comp in co_grp:
        min_date = comp[1]["dateRep"].min()
        max_date = comp[1]["dateRep"].max()
        nr_days = (max_date-min_date).days+1
        delta = nr_days-comp[1].shape[0]
        if delta!= 0:
            nr_countr+=1
            total_nr_days_missing += delta
            #un comment to print all affected countries
            #print(str(comp[0]) +" " + str(delta))
    return nr_countr, total_nr_days_missing
nr_countr, total_nr_days_missing = check_in_between_missing_values(ecdc_df)
print("\n affected nr countries: " + str(nr_countr))
print("total nr missing: " +str(total_nr_days_missing))


print("Attention : Cumulative_number_for_14_days_of_COVID-19_cases_per_100000 and cases_per_pop are not valid anymore(but also not used in the environment)")
print("\n Since there are many countries where the time series contain structure in weeks we filled missing values with zeros to keep that structure")


# %%
print("Furthermore there is no country that reported more than one datapoint for one day")

#check if there is a country having more than one entry per day -> NONE good!
for comp in co_grp:
    days = comp[1].groupby("dateRep")
    for day in days:
        if day[1].shape[0] != 1:
            print(comp[0] + " at: " + day[0])
print("DONE")


