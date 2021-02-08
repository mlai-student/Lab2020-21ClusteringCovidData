import pickle
import random
import logging

from sklearn.decomposition import PCA
from tslearn.utils import to_time_series_dataset
import pandas as pd
import numpy as np
from tslearn.metrics import dtw
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances

from src.data_generation.get_additional_info import get_additional_information_distance_functions


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

        self.additional_information_distance_functions = {}

    def fill_from_snippets(self, snippets, test_share=.1, data_gen_config=None):
        test_share = round(test_share * len(snippets))
        self.test_data = random.sample(snippets, test_share)
        self.train_data = [x for x in snippets if x not in self.test_data]
        self.n_examples = len(self.test_data) + len(self.train_data)

        if data_gen_config is not None:
            self.add_information_distance_functions = get_additional_information_distance_functions(data_gen_config)

    def concat_from_example(self, other):
        self.train_data.extend(other.train_data)
        self.test_data.extend(other.test_data)
        self.n_examples += other.n_examples
        return other.n_examples

    def to_ts_snippet(self):
        X_train = to_time_series_dataset([x.to_vector() for x in self.train_data])
        y_train = [x.label for x in self.train_data]
        X_test = y_test = None
        if len(self.test_data) > 0:
            X_test = to_time_series_dataset([x.to_vector() for x in self.test_data])
            y_test = [x.label for x in self.test_data]
        return X_train, X_test, y_train, y_test

    def split_examples(self):
        X_train = np.array([x.to_vector() for x in self.train_data])
        y_train = np.array([x.label for x in self.train_data])
        X_test = y_test = None
        if len(self.test_data) > 0:
            X_test = np.asarray([x.to_vector() for x in self.test_data])
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
        df = pd.DataFrame(cases_dict,
                          columns=['cases', 'countriesAndTerritories', 'countryterritoryCode', 'continentExp'])
        return df

    def divide_by_label(self, n_cluster, labels):
        cluster = [Examples() for _ in range(n_cluster)]
        for idx, l in enumerate(labels):
            if l != -1:
                data = self.train_data[idx]
                cluster[l].append_snippet(data)
        new_labels = []
        for i, c in enumerate(cluster):
            new_labels.extend([i for _ in range(c.n_examples)])
        n_per_clusters = [len(c.train_data) + len(c.test_data) for c in cluster]
        return cluster, n_per_clusters, new_labels

    def append_snippet(self, other_snippet):
        self.train_data.append(other_snippet)
        self.n_examples += 1

    def add_padding(self, cut_length=0):
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
            distance_matrix = euclidean_distances(np.asarray(X), np.asarray(X))
        else:
            return None
        return pairwise_distances(X=distance_matrix, metric='precomputed')

    def pca_reduction(self, n_components):
        pca = PCA(n_components=n_components)
        X = [x.to_vector() for x in self.train_data]
        pca.fit(X)
        X_pca = pca.transform(X)
        return X_pca

    def save_to_file(self, filename):
        try:
            pkl_file = open(filename, "wb")
            pickle.dump(self, pkl_file)
            pkl_file.close()
        except Exception as Argument:
            logging.error("Saving Example class failed with following message:")
            logging.error(str(Argument))

    def standardize(self):
        for t in self.train_data:
            t.standardize()
        for t in self.test_data:
            t.standardize()

    def resolve_standardize(self):
        for t in self.train_data:
            t.de_standardize()
        for t in self.test_data:
            t.de_standardize()
