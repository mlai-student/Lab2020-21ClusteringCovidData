import numpy as np

class Snippet:
    def __init__(self, ts, label):
        self.time_series = ts
        self.label = label
        self.temperature = self.country = self.continent = None

    def to_vector(self) -> np.array:
        features = list(filter(lambda x: x is not None,[self.temperature, self.country, self.continent]))
        return np.append(self.time_series, features)

    def smooth_time_series(self, method=""):
        pass
