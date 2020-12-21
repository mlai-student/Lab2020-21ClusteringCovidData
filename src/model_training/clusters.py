import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import logging
from datetime import date
from pathlib import Path
import pickle
import os
import plotly.express as px
import pandas as pd

import tslearn.clustering as ts

import sklearn.cluster as sk
import sklearn_extra.cluster as sk_extra

from src.data_representation.Examples import Examples


class GenericCluster:
    def fit(self, X: Examples):
        self.model.fit(self.preprocess(X))
        self.labels = self.model.labels_
        self.clusters, self.n_per_clusters = X.divide_by_label(self.n_clusters, labels=self.labels)
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

    def plot_geo_cluster(self):
        dfs = [c.make_dataframe() for c in self.clusters]
        df = pd.concat(dfs, ignore_index=True)
        df_acc = df.groupby(['countryterritoryCode'], as_index=False).sum()
        fig = px.choropleth(df_acc, locations="countryterritoryCode",
                            color=self.labels,
                            color_discrete_sequence=px.colors.qualitative.G10)
        fig.update_layout(
            title=f"Clustermethod: {self.name}, Number Clusters: {self.n_clusters}")
        return fig

    def save_model(self, filename, save_data=False, examples=None):
        try:
            PROJECT_PATH = os.getcwd().replace("notebooks", "") + "data/" + date.today().strftime("%b-%d-%Y")
            Path("{}/{}".format(PROJECT_PATH, "/model/")).mkdir(parents=True, exist_ok=True)
            # Path("{}/{}/{}".format(PROJECT_PATH, "/model/", filename)).mkdir(parents=True, exist_ok=True)
            with open(PROJECT_PATH + "/model/" + filename, 'wb') as f:
                pickle.dump(self, f)
            if save_data:
                with open(PROJECT_PATH + "/model", "wb") as pkl_file:
                    pickle.dump(examples, pkl_file)
        except Exception as Argument:
            logging.error("Saving model file failed with following message:")
            logging.error(str(Argument))

    def preprocess(self, X: Examples):
        pass

    def statistics(self):
        mu = sum([len(c.train_data) for c in self.clusters])/self.n_clusters
        var = sum([(c - mu)**2 for c in self.n_per_clusters])/self.n_clusters
        max_n_cluster = max(self.n_per_clusters)
        min_n_cluster = min(self.n_per_clusters)
        print(f"Statistic Report for: {self.name} with {self.n_clusters} different clusters")
        print(f"Expected Value: {mu} Variance: {var}\nBiggest Cluster: {max_n_cluster}, Smallest Cluster: {min_n_cluster}")



class KMedoids(GenericCluster):
    def __init__(self, n_clusters, metric):
        self.name = "KMedoids"
        if metric == "euclidean":
            self.model = sk_extra.KMedoids(n_clusters=n_clusters, metric='euclidean', random_state=42)
        else:
            self.model = sk_extra.KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
        self.metric = metric
        self.n_clusters = n_clusters

    def preprocess(self, X: Examples):
        if self.metric == "dtw":
            return X.to_distance_matrix(metric=self.metric)
        elif self.metric == "euclidean":
            X_train, X_test, y_train, y_test = X.split_examples()
            if X_test:
                return np.concatenate((X_train, X_test))
            else:
                return X_train
        return None


class KMeans(GenericCluster):
    def __init__(self, n_clusters, metric=None):
        self.name = "KMeans"
        self.model = sk.KMeans(n_clusters=n_clusters, random_state=42)
        self.n_clusters = n_clusters

    def preprocess(self, X: Examples):
        X_train, X_test, y_train, y_test = X.split_examples()
        if X_test:
            return np.concatenate((X_train, X_test))
        else:
            return X_train


class DBSCAN(GenericCluster):
    def __init__(self, eps, metric):
        self.name = "DBSCAN"
        if metric == "euclidean":
            self.model = sk.DBSCAN(eps=eps, metric='euclidean')
        else:
            self.model = sk.DBSCAN(eps=eps, metric='precomputed')
        self.metric = metric
        self.n_clusters = eps
        self.eps = eps

    def preprocess(self, X):
        if self.metric == "dtw":
            return X.to_distance_matrix(metric=self.metric)
        elif self.metric == "euclidean":
            X_train, X_test, y_train, y_test = X.split_examples()
            if X_test:
                return np.concatenate((X_train, X_test))
            else:
                return X_train
        return None

    def update_clusters(self):
        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)

class TS_KernelKMeans(GenericCluster):
    def __init__(self, n_clusters, metric=None):
        self.name = "TS_KernelKMeans"
        self.model = ts.KernelKMeans(n_clusters=n_clusters, kernel="gak", random_state=42)
        self.n_clusters = n_clusters

    def preprocess(self, X: Examples):
        X_train, X_test, y_train, y_test = X.to_ts_snippet()
        if X_test:
            return np.concatenate((X_train, X_test))
        else:
            return X_train


class TS_KShape(GenericCluster):
    def __init__(self, n_clusters, metric=None):
        self.name = "TS_KShape"
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
        self.name = "TS_KMeans"
        self.model = ts.TimeSeriesKMeans(n_clusters=n_clusters, metric=metric)
        self.n_clusters = n_clusters

    def preprocess(self, X: Examples):
        X_train, X_test, y_train, y_test = X.to_ts_snippet()
        if X_test:
            return np.concatenate((X_train, X_test))
        else:
            return X_train
