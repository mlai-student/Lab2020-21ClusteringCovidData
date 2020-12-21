from copy import deepcopy as dp
import numpy as np

#distord each value by a gaussian and add it to snippets
#no return
def data_augmentation(snippets, data_gen_config):
    #first distord each value by a gaussian with percent_varianz varianz
    distortet_snippets = []
    perc_dist = float(data_gen_config["percent_varianz"])
    for snippet in snippets:
        snippet_copy = dp(snippet)
        distortion = np.random.normal(0,perc_dist, len(snippet_copy.time_series))
        snippet_copy.time_series += np.multiply(snippet_copy.time_series, distortion)
        snippet_copy.label += snippet_copy.label*np.random.normal(0,perc_dist)
        distortet_snippets.append(snippet_copy)
    snippets.extend(distortet_snippets)
