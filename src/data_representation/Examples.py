import pickle
import random
import logging
from datetime import date
from pathlib import Path
from tslearn.utils import to_time_series_dataset
import pandas as pd
import numpy as np
from tslearn.metrics import dtw
from sklearn.metrics.pairwise import paired_euclidean_distances, pairwise_distances


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

    def fill_from_snippets(self, snippets, test_share=.1):
        test_share = round(test_share * len(snippets))
        self.test_data = random.sample(snippets, test_share)
        self.train_data = [x for x in snippets if x not in self.test_data]

    def to_ts_snippet(self):
        X_train = to_time_series_dataset([x.to_vector() for x in self.train_data])
        X_test = to_time_series_dataset([x.to_vector() for x in self.test_data])
        y_train = [to_time_series_dataset(x.label) for x in self.train_data]
        y_test = [to_time_series_dataset(x.label) for x in self.test_data]
        return X_train, X_test, y_train, y_test

    def split_examples(self):
        X_train = [x.to_vector() for x in self.train_data]
        X_test = [x.to_vector() for x in self.test_data]
        y_train = [x.label for x in self.train_data]
        y_test = [x.label for x in self.test_data]
        return X_train, X_test, y_train, y_test

    def reset_examples(self):
        self.train_data = []
        self.test_data = []

    def make_dataframe(self, use_test=False):
        ts_list = []
        country_list = []
        continent_list = []
        for snippet in self.train_data:
            length = snippet.time_series.size
            ts_list.extend(snippet.time_series)
            country_list.extend(([snippet.country] * length))
            continent_list.extend([snippet.continent] * length)
        cases_dict = {'cases': ts_list,
                'countriesAndTerritories': country_list,
                'continentExp': continent_list}
        df = pd.DataFrame(cases_dict, columns=['cases', 'countriesAndTerritories', 'continentExp'])
        return df

    def divide_by_label(self, n_cluster, labels):
        cluster = [Examples() for _ in range(n_cluster)]
        for idx, l in enumerate(labels):
            cluster[l].train_data.append(self.train_data[idx])
        return cluster

    def to_distance_matrix(self, metric='dtw'):
        if metric is 'dtw':
            distance = dtw
        elif metric is 'euclidean':
            distance = paired_euclidean_distances
        else:
            return None
        X = [ts.to_vector() for ts in self.train_data]
        size = len(self.train_data)
        distance_matrix = np.zeros(size, size)
        for i in range(size):
            for j in range(i, size):
                if i == j:
                    d = 0
                else:
                    d = dtw(X[i], X[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
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
