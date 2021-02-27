import numpy as np

class smooth_invert:
    def __init__(self, prev_days_smooth, shift):
        self.prev_days_smooth = prev_days_smooth
        self.shift = shift

    def invert(self, x):
        if isinstance(self.prev_days_smooth, list):
             self.prev_days_smooth = self.prev_days_smooth[0]
        return max((x - self.prev_days_smooth) / self.shift, 0)

# smooth a timeline X with a value Y which comes from a country timeline group and if from start to end
# average with mean over data_gen_config[nr_days_for_avg] days
# returning None if timeline is not smoothable
def smooth_timeline(X, Y, group_sort, start, end, data_gen_config, invert_functions, use_zero_filler=False, no_Y=False):
    nr_days_for_avg = int(data_gen_config["nr_days_for_avg"])
    # TODO add moore smoothing methods adjustable via config
    conv_matrix = np.array([1. / nr_days_for_avg for _ in range(nr_days_for_avg)])
    # first case if start is smaller than nr_days_for_avg we cant smooth so return a None
    if (not use_zero_filler) and start < nr_days_for_avg:
        return None
    X_out = []
    begin = max(start - nr_days_for_avg + 1, 0)
    smooth_data = group_sort.iloc[begin: end + 1].to_numpy()
    if start - nr_days_for_avg + 1 < 0:
        smooth_data = np.insert(smooth_data, 0, np.zeros(-(start - nr_days_for_avg + 1)))
    begin = nr_days_for_avg - 1
    end = len(smooth_data)
    if no_Y:
        end += 1
    for i in range(begin, end - 1):
        X_out.append(float(smooth_data[i - nr_days_for_avg + 1: i + 1].dot(conv_matrix)))
    if no_Y:
        return np.array(X_out)
    Y_out = float(smooth_data[end - nr_days_for_avg: end + 1].dot(conv_matrix))
    last_conv_entry = conv_matrix[-1] if conv_matrix[-1] != 0 else 0
    invert_obj = smooth_invert(float(smooth_data[end - nr_days_for_avg: end - 1].dot(conv_matrix[:-1])), last_conv_entry)
    invert_functions.insert(0, invert_obj)
    return np.array(X_out), np.array(Y_out)
