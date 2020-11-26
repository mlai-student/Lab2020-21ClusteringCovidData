'''
Collection of Time Series Algorithms from the tslearn package
'''

from tslearn.clustering import TimeSeriesKMeans, KernelKMeans

def KMeans(X, n_clusters, metric):
    km = TimeSeriesKMeans(n_clusters, metric)
    return km.fit_predict(X)

def KernelMeans(X, n_clusters, kernel):
    gak_km = KernelKMeans(n_clusters, kernel)
    return gak_km.fit_predict(X)


from tslearn.neighbors import KNeighborsTimeSeries

def KNeighbors(X, n_neighbors):
    knn = KNeighborsTimeSeries(n_neighbors)
    return knn.fit(X)


from tslearn.barycenters import dtw_barycenter_averaging

def Barycenters(X, barycenter_size):
    return dtw_barycenter_averaging(X, barycenter_size)
