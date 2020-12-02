import pickle
import random
import logging
from datetime import date
from pathlib import Path
from tslearn.utils import to_time_series_dataset

def load_Examples_from_file(filename):
    #read the pickle file
    pkl_file = open(filename, 'rb')
    #unpickle the dataframe
    output = pickle.load(pkl_file)
    #close file
    pkl_file.close()
    return output

class Examples:

    def __init__(self):
        self.train_examples = []
        self.test_examples = []

    def fill_from_snippets(self, snippets, test_share=.1):
        test_share = round(test_share*len(snippets))
        self.test_examples = random.sample(snippets, test_share)
        self.train_examples = [x for x in snippets if x not in self.test_examples]


    def make_ts_snippet(self):
        X_train = to_time_series_dataset([x.to_vector() for x in self.train_examples])
        X_test = to_time_series_dataset([x.to_vector() for x in self.test_examples])
        y_train = [x.label for x in self.train_examples]
        y_test = [x.label for x in self.test_examples]
        return X_train, X_test, y_train, y_test

    def reset_examples(self):
        self.train_examples = []
        self.test_examples = []

    def save_to_file(self, filename):
        try:
            today = date.today().strftime("%b-%d-%Y")
            Path("data/" + today).mkdir(parents=True, exist_ok=True)
            pkl_file = open("data/{}/{}".format(today, filename), "wb")
            pickle.dump(self, pkl_file)
            pkl_file.close()
        except Exception as Argument:
            logging.error("Saving Example class failed with following message:")
            logging.error(str(Argument))
