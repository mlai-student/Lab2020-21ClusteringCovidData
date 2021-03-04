import logging, pickle, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.cluster as sk
import sklearn_extra.cluster as sk_extra
import tslearn.clustering as ts
from sklearn.exceptions import NotFittedError
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from src.data_representation.Examples import Examples
from src.data_representation.config_to_dict import get_config_dict

class GenericCluster:
    def fit(self, X: Examples):
        self.model.fit(self.preprocess(X))
        self.clusters, self.n_per_clusters, self.labels = X.divide_by_label(self.n_clusters, labels=self.model.labels_)
        return self

    def predict(self, X_test: Examples):
        try:
            X_test = self.preprocess(X_test, predict=True)
            return self.model.predict(X_test)
        except NotFittedError as e:
            print(repr(e))

    def plot_cluster(self):
        fig, axs = plt.subplots(nrows=self.n_clusters, figsize=(10, self.n_clusters * 2))
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(self.clusters))))
        for i, cl in enumerate(self.clusters):
            cl_color = next(color)
            for _ts in cl.train_data:
                axs[i].plot(_ts.time_series, c=cl_color)
                axs[i].set_title(str(len(cl.train_data)))
        # fig.suptitle('Clusters')
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

    def save_model_and_update_overview(self, main_config, filename):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        except Exception as Argument:
            logging.error("Saving model file failed with following message:")
            logging.error(str(Argument))
        print('saving works')
        #updating main cfg file:
        self.add_model_entry_to_overview_csv(main_config, filename)

    def add_model_entry_to_overview_csv(self, main_config, filename_model):
        cfg_settings_dict = get_config_dict(main_config)
        cfg_settings_dict["filename_model"] = filename_model
        #add cluster scores to the output table
        cfg_settings_dict["Silhouette Score"] = self.silhouette(metric = main_config["model_training_settings"]["metric"])
        cfg_settings_dict["Calinski Score"] = self.calinski()
        cfg_settings_dict["Davies Score"] = self.davies()
        cfg_settings_dict["Cluster size variance"],_,_ = self.statistics()

        foldername = main_config["data_generating_settings"]["generated_folder_path"]
        csv_filename = foldername + "model_overview.csv"
        if os.path.isfile(csv_filename):
            df = pd.read_csv(csv_filename)
            pd_dict = pd.Series(cfg_settings_dict).to_frame().T
            model_df = df.append(pd_dict.iloc[0])
        else:
            model_df = pd.Series(cfg_settings_dict).to_frame().T
        model_df.to_csv(csv_filename, index=False)

    def preprocess(self, X: Examples, predict=False):
        pass

    def get_examples_from_cluster(self):
        X = Examples()
        for ex in self.clusters:
            X.concat_from_example(ex)
        return X

    def silhouette(self, metric=None, use_add_info=False, key="Population"):
        if metric == "dtw":
            from tslearn.clustering import silhouette_score
        else:
            from sklearn.metrics import silhouette_score
        try:
            X = self.get_examples_from_cluster()
            if use_add_info:
                X_add_info = np.array([sn.additional_info[key] for sn in X.train_data]).reshape(-1, 1)
                return silhouette_score(X_add_info, labels=self.labels,
                                        metric="l1", random_state=42)
            else:
                if metric == "dtw":
                    X_train, X_test, y_train, y_test = X.to_ts_snippet()
                    return silhouette_score(X_train, labels=self.labels,
                                            metric="dtw", random_state=42)
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
        max_n_cluster, min_n_cluster = max(self.n_per_clusters), min(self.n_per_clusters)
        if verbose:
            print(f"Statistic Report for: {self.name} with {self.n_clusters} different clusters")
            print(f"Expected Value: {mu} Variance: {var}\nBiggest Cluster: {max_n_cluster}, Smallest Cluster: {min_n_cluster}")
        return var, max_n_cluster, min_n_cluster


class KMedoids(GenericCluster):
    def __init__(self, n_clusters, metric):
        self.name = "KMedoids"
        if metric == "euclidean":
            self.model = sk_extra.KMedoids(n_clusters=n_clusters, metric='euclidean', random_state=42)
        else:
            self.model = sk_extra.KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
        self.metric, self.n_clusters = metric, n_clusters

    def preprocess(self, X: Examples, predict=False):
        if self.metric == "euclidean":
            X_train, X_test, y_train, y_test = X.split_examples()
            return X_test if predict else X_train
        return None


class KMeans(GenericCluster):
    def __init__(self, n_clusters, metric=None):
        self.name = "KMeans"
        self.model = sk.KMeans(n_clusters=n_clusters, random_state=42)
        self.n_clusters, self.metric = n_clusters, "euclidean"

    def preprocess(self, X: Examples, predict=False):
        X_train, X_test, y_train, y_test = X.split_examples()
        return X_test if predict else X_train


class DBSCAN(GenericCluster):
    def __init__(self, eps, metric):
        self.name = "DBSCAN"
        if metric == "euclidean":
            self.model = sk.DBSCAN(eps=eps, metric='euclidean', min_samples=4)
        else:
            self.model = sk.DBSCAN(eps=eps, metric='precomputed')
        self.metric, self.n_clusters, self.eps = metric, eps, eps

    def preprocess(self, X, predict=False):
        if self.metric == "dtw":
            return X.to_distance_matrix(metric=self.metric)
        elif self.metric == "euclidean":
            X_train, X_test, y_train, y_test = X.split_examples()
            return X_test if predict else X_train
        return None

    def update_clusters(self):
        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)


class TS_KernelKMeans(GenericCluster):
    def __init__(self, n_clusters, metric=None):
        self.name = "TS_KernelKMeans"
        self.model = ts.KernelKMeans(n_clusters=n_clusters, kernel="gak", random_state=42)
        self.n_clusters, self.metric = n_clusters, "dtw"

    def preprocess(self, X: Examples, predict=False):
        X_train, X_test, y_train, y_test = X.to_ts_snippet()
        return X_test if predict else X_train


class TS_KShape(GenericCluster):
    def __init__(self, n_clusters, metric=None):
        self.name = "TS_KShape"
        self.model = ts.KShape(n_clusters=n_clusters)
        self.n_clusters, self.metric = n_clusters, "dtw"

    def preprocess(self, X: Examples, predict=False):
        X_train, X_test, y_train, y_test = X.to_ts_snippet()
        return X_test if predict else X_train


# Carefull, implements a
class TS_KMeans(GenericCluster):
    def __init__(self, n_clusters, metric):
        self.name = "TS_KMeans"
        self.model = ts.TimeSeriesKMeans(n_clusters=n_clusters, metric=metric)
        self.n_clusters, self.metric = n_clusters, metric

    def preprocess(self, X: Examples, predict=False):
        X_train, X_test, y_train, y_test = X.to_ts_snippet()
        return X_test if predict else X_train
