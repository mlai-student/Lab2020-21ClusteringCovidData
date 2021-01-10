import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import numpy as np
import logging
from datetime import date
import os
from pathlib import Path
import pickle
import plotly.express as px
import pandas as pd
import tslearn.clustering as ts
import sklearn.cluster as sk
import sklearn_extra.cluster as sk_extra
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from src.data_representation.Examples import Examples


class GenericCluster:
    def fit(self, X: Examples):
        self.model.fit(self.preprocess(X))
        self.clusters, self.n_per_clusters, self.labels = X.divide_by_label(self.n_clusters, labels=self.model.labels_)
        return self

    def plot_cluster(self):
        fig, axs = plt.subplots(self.n_clusters, figsize=(12, self.n_clusters * 3))
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
        labels = [str(l) for l in self.labels]
        fig = px.choropleth(df_acc, locations="countryterritoryCode",
                            color=labels)
        fig.update_layout(
            title=f"Clustermethod: {self.name}, Number Clusters: {self.n_clusters}")
        return fig

    def save_model(self, data_path, filename, save_data=False, examples=None):
        try:
            # PROJECT_PATH = os.getcwd().replace("notebooks", "") + "data/" + date.today().strftime("%b-%d-%Y")
            Path("{}/{}".format(data_path, "/model/")).mkdir(parents=True, exist_ok=True)
            # Path("{}/{}/{}".format(PROJECT_PATH, "/model/", filename)).mkdir(parents=True, exist_ok=True)
            with open(data_path + "/model/" + filename, 'wb') as f:
                pickle.dump(self, f)
            if save_data:
                with open(data_path + "/model", "wb") as pkl_file:
                    pickle.dump(examples, pkl_file)
        except Exception as Argument:
            logging.error("Saving model file failed with following message:")
            logging.error(str(Argument))

    def preprocess(self, X: Examples):
        pass

    def get_examples_from_cluster(self):
        X = Examples()
        for ex in self.clusters:
            X.concat_from_example(ex)
        return X

    def silhouette(self, metric=None, use_add_info=False, key="Population"):
        try:
            X = self.get_examples_from_cluster()
            if use_add_info:
                X_add_info = np.array([sn.additional_info[key] for sn in X.train_data]).reshape(-1, 1)
                return silhouette_score(X_add_info, labels=self.labels,
                                        metric="l1", random_state=42)
            else:
                if metric == "dtw":
                    return silhouette_score(X.to_distance_matrix(metric="dtw"), labels=self.labels,
                                            metric="precomputed", random_state=42)
                else:
                    X_train, _, _, _ = X.split_examples()
                    return silhouette_score(X_train, labels=self.labels,
                                            metric="euclidean", random_state=42)

        except Exception as Argument:
            logging.error("Computing silhouette score failed with following message:")
            logging.error(str(Argument))

    def calinski(self, use_add_info=False, key="Population"):
        try:
            X = self.get_examples_from_cluster()
            if use_add_info:
                X_add_info = np.array([sn.additional_info[key] for sn in X.train_data]).reshape(-1, 1)
                return calinski_harabasz_score(X_add_info, labels=self.labels)
            else:
                X_train, _, _, _ = X.split_examples()
                return calinski_harabasz_score(X_train, labels=self.labels)
        except Exception as Argument:
            logging.error("Computing silhouette score failed with following message:")
            logging.error(str(Argument))

    def davies(self, use_add_info=False, key="Population"):
        try:
            X = self.get_examples_from_cluster()
            if use_add_info:
                X_add_info = np.array([sn.additional_info[key] for sn in X.train_data]).reshape(-1, 1)
                return davies_bouldin_score(X_add_info, labels=self.labels)
            else:
                X_train, _, _, _ = X.split_examples()
                return davies_bouldin_score(X_train, labels=self.labels)

        except Exception as Argument:
            logging.error("Computing silhouette score failed with following message:")
            logging.error(str(Argument))

    def statistics(self, verbose=False):
        mu = sum([len(c.train_data) for c in self.clusters]) / self.n_clusters
        var = np.var(self.n_per_clusters)
        max_n_cluster = max(self.n_per_clusters)
        min_n_cluster = min(self.n_per_clusters)
        if verbose:
            print(f"Statistic Report for: {self.name} with {self.n_clusters} different clusters")
            print(
                f"Expected Value: {mu} Variance: {var}\nBiggest Cluster: {max_n_cluster}, Smallest Cluster: {min_n_cluster}")
        return var, max_n_cluster, min_n_cluster


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
        self.metric = "euclidean"

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
        self.metric = "dtw"

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
        self.metric = "dtw"

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
        self.metric = metric

    def preprocess(self, X: Examples):
        X_train, X_test, y_train, y_test = X.to_ts_snippet()
        if X_test:
            return np.concatenate((X_train, X_test))
        else:
            return X_train
