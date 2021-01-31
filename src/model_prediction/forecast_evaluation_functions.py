#TODO Varianz einfuegen wenn benötigt
#gettign a list of snippets
def avg_perc_dist(forecast_snippet_list):
    avg_perc_dist_sum = 0
    samples_count = 0
    for snippet in forecast_snippet_list:
        inverted_label = snippet.invert_to_abs_cases(snippet.label)
        if inverted_label != 0:
            inverted_forecast = snippet.invert_to_abs_cases(snippet.forecast)

            avg_perc_dist_sum += abs(inverted_forecast-inverted_label)/inverted_label
            samples_count +=1
    return avg_perc_dist_sum/samples_count
