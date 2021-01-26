
#gettign a list of snippets
def avg_perc_dist(forecast_snippet_list):
    avg_perc_dist_sum = 0
    samples_count = 0
    for snippet in forecast_snippet_list:
        if snippet.label != 0:
            avg_perc_dist_sum += abs(snippet.forecast-snippet.label)/snippet.label
            samples_count +=1
    return avg_perc_dist_sum/samples_count
