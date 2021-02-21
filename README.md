# Lab2020-21ClusteringCovidData
Lab project on Clustering Time Series

---
## Motivation

---
## Key Features
This projects' framework provides following functionalities:
* **Data generation** - Use a current edition of the ECDC dataset or a local copy. Do pre-proccessing and choose cluster length and possibly country split for training and testing.
* **Data representation** - Example and Snippet class to be flexible with regards to the given dataset
* **Clustering** - Clustering methods based on sklearn and tslearn librarys
* **Forecasting** - Naive forecasting methods as well as a LSTM neural network
* **Result** - Clustering and forecasting results as well as trained models will be saved
---

## Getting Started

1. Clone this repository,
2. Navigate to the directory and `pip install -r requirements.txt` to get all requirements
3. For executing the framework with default configuration use `python run.py --path_to_cfg config.ini`

## Config.ini
The configuration file holds all important parameters that can be tuned for a run through the framework.

* **Main flow settings** control what parts of the pipeline are turned on/off and where you'd like to save your results.
* **Data generating settings** lets you specify the dataset to use, the name indicating the column containing the daily/weekly case data, if you'd like to use complete time series or a subsample of specific length, the number of days to predict and a country test share for separate training and testing. For the pre-processing data smoothing and preprocessing can be switched on/off.
* **Model training settings** holds what clustering methods, how many clusters and what distance metrics are used.
* Lastly in **Model prediction settings** the forecasting method and evaluation function are chosen.

## Cluster Methods
1. **KMedoids** (euclidean)
2. **KMeans** (euclidean/dtw)
3. **DBSCAN** (euclidean/(slow) dtw support)
4. **KernelMeans** (dtw)
5. **KShape** (dtw)

Additionally these **scoring methods** are included: Silhouette score, Calinski, Davies (link), variance measure

## Forecasting Methods
1. **Naive forecast**
2. **Seasonal naive forecast**
3. **LSTM** (with and without clustering)
---

## Visualization
