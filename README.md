# Lab2020-21ClusteringCovidData
Lab project on Clustering Time Series

---
## Motivation

The corona virus dictates how many people have to live their everyday life. Finding similar case developments between countries and forecasting Covid-19 cases may help in overcoming the pandemic by acting foresighted. For this purpose, we developed a framework that is capable of clustering time-series data of Covid-19 cases and predicting future cases using the results.
---
## Key Features
This projects' framework provides following functionalities:
* **Data generation** - Use a current edition of the ECDC dataset or a local copy. Do pre-proccessing and choose cluster length and possibly country split for training and testing. Result is a pickle file of an Examples class containing all the generated Time Serieses (Snippets or whole country Series). Those form the data basis for Clustering and Forecasting.
* **Data representation** - Example and Snippet class. Those are the Container for the Timeseries Data. (A Example class contains many Snippets)
* **Clustering** - Clustering methods based on sklearn and tslearn librarys. Clustering the Snippets in an Examples file and saving that inforamtion in an correpsonding output file in the "models/" folder.
* **Forecasting** - Naive forecasting methods as well as a LSTM neural network. Using the data provided by an Examples (pickle) file (and if needed a models/ file
* **Result** - Clustering and forecasting results as well as trained models will be saved
* **Workflow** - Define in config.ini and variables_cfg.ini all settings to be tested (e.g. Multiple different Clustering methods). For each possible combination of those settings a Dataset is beeing generated, as well as Clustering models are trained and the desired Forecast method will be applied and evaluated. The results of each possible combination gets clearly presented using csv files (one for Data Generation, Model training and Forecast results each). Those CSV files show the filenames with the corresponding setting they were created and corresponding results (e.g. For Forecasting). 
*  **Analysation/Visualization** - To analyse the outputs for each combination there are several notebook online to read visualize e.g. the Clustering output. 
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
