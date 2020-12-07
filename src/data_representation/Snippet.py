import numpy as np

class Snippet:
    def __init__(self, ts, label, country=None, continent=None, ascending=False):
        self.time_series = ts
        if ascending:
            self.time_series = np.flipud(self.time_series)
        self.label = label
        self.temperature = None
        self.country = country
        self.continent = continent

    def to_vector(self, only_ts=True) -> np.array:
        if only_ts:
            return self.time_series
        else:
            features = list(filter(lambda x: x is not None,[self.temperature, self.country, self.continent]))
            return np.append(self.time_series, features)

    def smooth_time_series(self, method=""):
        pass
    
    def weigh_ts(self):
        pass
