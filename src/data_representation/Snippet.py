import numpy as np

class Snippet:
    def __init__(self, ts: np.array, label, country_id = None, country=None,
                continent=None, flip_order=False, additional_info={}):
        self.time_series = ts
        if flip_order:
            self.time_series = np.flipud(self.time_series)
        self.label = label
        self.forecast = 0
        self.country_id = country_id
        self.country = country
        self.continent = continent
        #dict to store additional info for snippet -> for each label (temp etc) should be an corresponding distance function saved in the corresproding Examples file
        self.additional_info = additional_info

    def to_vector(self, only_ts=True) -> np.array:
        if only_ts:
            return self.time_series
        else:
            features = list(filter(lambda x: x is not None,[self.temperature, self.country, self.continent]))
            return np.append(self.time_series, features)
