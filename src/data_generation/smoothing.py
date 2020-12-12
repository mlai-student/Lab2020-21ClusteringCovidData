import numpy as np
#smooth a timeline X with a value Y which comes from a coutry timline group and if from start to end
#average with mean over data_gen_config[nr_days_for_avg] days
#returning None if timeline is not smoothable
def smooth_timeline(X, Y, group_sort, start, end, data_gen_config):
    nr_days_for_avg = int(data_gen_config["nr_days_for_avg"])

    #first case if start is smaller than nr_days_for_avg we cant smooth so return a None
    if start < nr_days_for_avg:
        return None
    X_out = []
    for i in range(start, end):
        X_out.append(round(float(group_sort.iloc[i-nr_days_for_avg+1: i+1].mean())))
    Y_out = round(float(group_sort.iloc[end+1-nr_days_for_avg+1: end+1+1].mean()))
    return np.array(X_out), Y_out
