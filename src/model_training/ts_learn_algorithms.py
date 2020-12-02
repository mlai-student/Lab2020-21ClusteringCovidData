'''
Collection of Time Series Algorithms from the tslearn package
'''

from tslearn.clustering import TimeSeriesKMeans, KernelKMeans


def kMeans(X, n_clusters, m):
    km = TimeSeriesKMeans(n_clusters, metric=m)
    return km.fit(X)


def kernelMeans(X, no, k):
    gak_km = KernelKMeans(n_clusters=no, kernel=k)
    return gak_km.fit(X)


from tslearn.neighbors import KNeighborsTimeSeries


def kNeighbors(X, n_neighbors):
    knn = KNeighborsTimeSeries(n_neighbors)
    return knn.fit(X)

# from tslearn.barycenters import dtw_barycenter_averaging

# def Barycenters(X, barycenter_size):
#     dtw_barycenter_averaging(X, barycenter_size)
