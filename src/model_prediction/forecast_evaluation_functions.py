import numpy as np
import matplotlib.pyplot as plt

#TODO Varianz einfuegen wenn benötigt
#gettign a list of snippets
def avg_perc_dist(forecast_snippet_list):
    avg_perc_dist_sum = 0
    samples_count = 0
    for snippet in forecast_snippet_list:
        inverted_label = round(snippet.invert_to_abs_cases(snippet.label))
        #if inverted_label <1000:
        #    continue
        if inverted_label < 0:
            print(f"Snippet label: {snippet.label} Snippet forecast {snippet.forecast} and inverted label {inverted_label}")
        if inverted_label != 0:
            inverted_forecast = round(snippet.invert_to_abs_cases(snippet.forecast))
            if inverted_forecast < 0:
                print(f"Snippet label: {snippet.label} Snippet forecast {snippet.forecast} and inverted forecast {inverted_forecast}")


            # if abs(round(inverted_forecast)-round(inverted_label))/round(inverted_label) > 10:
            #     print(f"hier zu hoch: orignal: {snippet.label}, {snippet.forecast}")
            #     print(f"inverted: {inverted_label}, {inverted_forecast}")
            #     print(f"original label {snippet.original_label}")
            #     print([round(snippet.invert_to_abs_cases(0.1*i),2) for i in range(10)])

            avg_perc_dist_sum += abs(inverted_forecast-inverted_label)/inverted_label
            samples_count += 1
    return avg_perc_dist_sum/samples_count


def histogram_perc_dist(forecast_snippet_list):
    perc_dists = []
    samples_count = 0
    for snippet in forecast_snippet_list:
        inverted_label = snippet.invert_to_abs_cases(snippet.label)
        if inverted_label != 0:
            inverted_forecast = snippet.invert_to_abs_cases(snippet.forecast)
            perc_dists.append(abs(inverted_forecast-inverted_label)/inverted_label)
            samples_count += 1
    plt.hist(perc_dists, density=True, bins=10)
    plt.show()

    return sum(np.array(perc_dists) / samples_count)
