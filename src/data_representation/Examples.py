import pickle
import random
import logging
from datetime import date
from pathlib import Path
from tslearn.utils import to_time_series_dataset
import pandas as pd
import numpy as np
from tslearn.metrics import dtw
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances


def load_Examples_from_file(filename):
    # read the pickle file
    pkl_file = open(filename, 'rb')
    # unpickle the dataframe
    output = pickle.load(pkl_file)
    # close file
    pkl_file.close()
    return output


class Examples:

    def __init__(self):
        self.train_data = []
        self.test_data = []
        self.n_examples = 0

    def fill_from_snippets(self, snippets, test_share=.1):
        test_share = round(test_share * len(snippets))
        self.test_data = random.sample(snippets, test_share)
        self.train_data = [x for x in snippets if x not in self.test_data]
        self.n_examples = len(self.test_data) + len(self.train_data)

    def to_ts_snippet(self):
        X_train = to_time_series_dataset([x.to_vector() for x in self.train_data])
        y_train = [x.label for x in self.train_data]
        X_test = y_test = []
        if len(self.test_data) > 0:
            X_test = to_time_series_dataset([x.to_vector() for x in self.test_data])
            y_test = [x.label for x in self.test_data]
        return X_train, X_test, y_train, y_test

    def split_examples(self):
        X_train = np.array([x.to_vector() for x in self.train_data])
        y_train = np.array([x.label for x in self.train_data])
        X_test = y_test = None
        if len(self.test_data) > 0:
            X_test = np.array([x.to_vector() for x in self.test_data])
            y_test = np.array([x.label for x in self.test_data])
        return X_train, X_test, y_train, y_test

    def reset_examples(self):
        self.train_data = []
        self.test_data = []

    def make_dataframe(self, use_test=False):
        ts_list = []
        country_list = []
        continent_list = []
        country_id_list = []
        for snippet in self.train_data:
            length = snippet.time_series.size
            ts_list.extend(snippet.time_series)
            country_list.extend(([snippet.country] * length))
            country_id_list.extend(([snippet.country_id] * length))
            continent_list.extend([snippet.continent] * length)
        cases_dict = {'cases': ts_list,
                'countriesAndTerritories': country_list,
                'countryterritoryCode': country_id_list,
                'continentExp': continent_list}
        df = pd.DataFrame(cases_dict, columns=['cases', 'countriesAndTerritories', 'countryterritoryCode', 'continentExp'])
        return df

    def divide_by_label(self, n_cluster, labels) -> list:
        cluster = [Examples() for _ in range(n_cluster)]
        for idx, l in enumerate(labels):
            data = self.train_data[idx]
            cluster[l].train_data.append(data)
        n_per_clusters = [len(c.train_data) + len(c.test_data) for c in cluster]
        return cluster, n_per_clusters

    def add_padding(self):
        ts_size = [ts.time_series.shape[0] for ts in self.train_data]
        ts_size.extend([ts.time_series.shape[0] for ts in self.test_data])
        max_ts_size = max(ts_size)
        for ts in self.train_data:
            n_zeros = max_ts_size - ts.time_series.shape[0]
            zeros = np.zeros(n_zeros)
            ts.time_series = np.concatenate((zeros, ts.time_series))
        for ts in self.test_data:
            n_zeros = max_ts_size - ts.time_series.shape[0]
            zeros = np.zeros(n_zeros)
            ts.time_series = np.concatenate((zeros, ts.time_series))

    def to_distance_matrix(self, metric='dtw'):
        X = [ts.to_vector() for ts in self.train_data]
        size = len(X)
        distance_matrix = np.zeros(shape=(size, size))
        if metric is 'dtw':
            for i in range(size):
                for j in range(i, size):
                    if i == j:
                        d = 0
                    else:
                        d = dtw(X[i], X[j])
                    distance_matrix[i, j] = d
                    distance_matrix[j, i] = d
        elif metric is 'euclidean':
            distance_matrix = euclidean_distances(X, X)
        else:
            return None
        return pairwise_distances(X=distance_matrix, metric='precomputed')

    def save_to_file(self, filename):
        try:
            today = date.today().strftime("%b-%d-%Y")
            Path("data/" + today).mkdir(parents=True, exist_ok=True)
            pkl_file = open("data/{}/{}".format(today, filename), "wb")
            pickle.dump(self, pkl_file)
            pkl_file.close()
        except Exception as Argument:
            logging.error("Saving Example class failed with following message:")
            logging.error(str(Argument))
