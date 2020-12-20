import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import logging
from datetime import date
from pathlib import Path
import pickle
import os

import tslearn.clustering as ts

import sklearn.cluster as sk
import sklearn_extra.cluster as sk_extra

from src.data_representation.Examples import Examples


class GenericCluster:
    def fit(self, X):
        self.model.fit(self.preprocess(X))
        self.labels = self.model.labels_
        self.clusters = X.divide_by_label(self.n_clusters, labels=self.labels)
        return self

    def plot_cluster(self):
        fig, axs = plt.subplots(self.n_clusters, figsize=(12,self.n_clusters*3))
        fig.suptitle('Clusters')

        for i, cl in enumerate(self.clusters):
            color = iter(cm.rainbow(np.linspace(0, 1, len(cl.train_data))))
            for ts in cl.train_data:
                cl_color = next(color)
                axs[i].plot(ts.time_series, c=cl_color)
                axs[i].set_title(str(len(cl.train_data)))
        return plt

    def save_model(self, save_data=False, examples=None):
        try:
            today = date.today().strftime("%b-%d-%Y")
            Path("data/" + today).mkdir(parents=True, exist_ok=True)
            Path("data/{}/{}".format(today, "model")).mkdir(parents=True, exist_ok=True)
            self.model.to_pickle("data/{}/{}/{}".format(today, "model", str(self.model.__class__.__name__)))
            if save_data:
                with open("data/{}/{}/{}_{}".format(today, "model",
                                                    str(self.model.__class__.__name__), "data"), "wb") as pkl_file:
                    pickle.dump(examples, pkl_file)
        except Exception as Argument:
            logging.error("Saving model file failed with following message:")
            logging.error(str(Argument))

    def preprocess(self, X: Examples):
        pass


class KMedoids(GenericCluster):
    def __init__(self, n_clusters, metric):
        self.model = sk_extra.KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
        self.metric = metric
        self.n_clusters = n_clusters

    def preprocess(self, X: Examples):
        return X.to_distance_matrix(metric=self.metric)


class KMeans(GenericCluster):
    def __init__(self, n_clusters):
        self.model = sk.KMeans(n_clusters=n_clusters, random_state=42)
        self.n_clusters = n_clusters

    def preprocess(self, X: Examples):
        X_train, X_test, y_train, y_test = X.split_examples()
        if X_test:
            return np.concatenate((X_train, X_test))
        else:
            return X_train


class Agglomerative(GenericCluster):
    def __init__(self, n_clusters, metric):
        self.model = sk.AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage="average")

    def preprocess(self, X):
        if self.metric == ["dtw", "euclidean"]:
            return X.to_distance_matrix(metric=self.metric)
        return None


class DBSCAN(GenericCluster):
    def __init__(self, eps, metric):
        self.model = sk.DBSCAN(eps=eps, metric="precomputed")
        self.metric = metric

    def preprocess(self, X: Examples):
        if self.metric == ["dtw", "euclidean"]:
            return X.to_distance_matrix(metric=self.metric)
        return None


class TS_KernelKMeans(GenericCluster):
    def __init__(self, n_clusters):
        self.model = ts.KernelKMeans(n_clusters=n_clusters, kernel="gak", random_state=42)
        self.n_clusters = n_clusters

    def preprocess(self, X: Examples):
        X_train, X_test, y_train, y_test = X.to_ts_snippet()
        if X_test:
            return np.concatenate((X_train, X_test))
        else:
            return X_train


class TS_KShape(GenericCluster):
    def __init__(self, n_clusters):
        self.model = ts.KShape(n_clusters=n_clusters)
        self.n_clusters = n_clusters

    def preprocess(self, X: Examples):
        X_train, X_test, y_train, y_test = X.to_ts_snippet()
        if X_test:
            return np.concatenate((X_train, X_test))
        else:
            return X_train


class TS_KMeans(GenericCluster):
    def __init__(self, n_clusters, metric):
        self.model = ts.TimeSeriesKMeans(n_clusters=n_clusters, metric=metric)
        self.n_clusters = n_clusters

    def preprocess(self, X: Examples):
        X_train, X_test, y_train, y_test = X.to_ts_snippet()
        if X_test:
            return np.concatenate((X_train, X_test))
        else:
            return X_train
