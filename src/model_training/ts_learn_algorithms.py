from datetime import date
from pathlib import Path
import logging


def save_time_series_model(model):
    try:
        today = date.today().strftime("%b-%d-%Y")
        Path("data/" + today).mkdir(parents=True, exist_ok=True)
        Path("data/{}/{}".format(today, "model") ).mkdir(parents=True, exist_ok=True)
        model.to_pickle("data/{}/{}/{}".format(today, "model" ,str(model.__class__.__name__)))
        
    except Exception as Argument:
        logging.error("Saving model file failed with following message:")
        logging.error(str(Argument))
    
    

'''
Collection of Time Series Algorithms from the tslearn package
'''

from tslearn.clustering import TimeSeriesKMeans, KernelKMeans

def KMeans(X, n_clusters, metric):
    km = TimeSeriesKMeans(n_clusters, metric)
    km.fit_predict(X)
    save_time_series_model(km)

def KernelMeans(X, n_clusters, kernel):
    gak_km = KernelKMeans(n_clusters, kernel)
    gak_km.fit_predict(X)
    save_time_series_model(gak_km)


from tslearn.neighbors import KNeighborsTimeSeries

def KNeighbors(X, n_neighbors):
    knn = KNeighborsTimeSeries(n_neighbors)
    knn.fit(X)
    save_time_series_model(knn)


# from tslearn.barycenters import dtw_barycenter_averaging

# def Barycenters(X, barycenter_size):
#     dtw_barycenter_averaging(X, barycenter_size)


