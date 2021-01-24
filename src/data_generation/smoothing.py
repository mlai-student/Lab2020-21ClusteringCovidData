import numpy as np
#smooth a timeline X with a value Y which comes from a coutry timline group and if from start to end
#average with mean over data_gen_config[nr_days_for_avg] days
#returning None if timeline is not smoothable
def smooth_timeline(X, Y, group_sort, start, end, data_gen_config, use_zero_filler=False, no_Y=False):
    nr_days_for_avg = int(data_gen_config["nr_days_for_avg"])
    #TODO add moore smoothing methods adjustable via config
    conv_matrix = np.array([1./nr_days_for_avg for _ in range(nr_days_for_avg)])
    #first case if start is smaller than nr_days_for_avg we cant smooth so return a None
    if (not use_zero_filler) and start < nr_days_for_avg:
        return None
    X_out = []
    begin = max(start-nr_days_for_avg+1 ,0)
    smooth_data = group_sort.iloc[begin: end+1].to_numpy()
    if start-nr_days_for_avg+1 <0:
        smooth_data = np.insert(smooth_data, 0, np.zeros(-(start-nr_days_for_avg+1)))
    begin = nr_days_for_avg-1
    end = len(smooth_data)
    if no_Y:
        end+=1
    for i in range(begin, end-1):
        X_out.append(float(smooth_data[i-nr_days_for_avg+1: i+1].dot(conv_matrix)))
    if no_Y:
        return np.array(X_out)
    Y_out = float(smooth_data[end-nr_days_for_avg: end+1].dot(conv_matrix))
    return np.array(X_out), np.array(Y_out)
