# TODO: Extract datageneration and move to data_generating
class Examples:

    def __init__(self, df=None, date=None, search_value=[], group=None):
        self.train_examples = []
        self.test_examples = []


    def make_examples_from_snippets(self, snippets, test_share=.1):
        test_share = round(test_share*len(snippets))
        self.test_examples = random.sample(snippets, test_share)
        self.train_examples = [x for x in snippets if x not in self.test_examples]


    def make_ts_snippet(self):
        X_train = to_time_series_dataset([x.to_vector() for x in self.train_examples])
        X_test = to_time_series_dataset([x.to_vector() for x in self.test_examples])
        y_train = [x.label for x in self.train_examples]
        y_test = [x.label for x in self.test_examples]
        return X_train, X_test, y_train, y_test


    def transform_df(self):
        pass


    def manipulate_df(self):
        pass


    def update_groups(self, date=None, group=None):
    self.groups = self.df[[self.date, self.group, self.search_val]].groupby(self.group)


    def reset_examples(self):
        self.train_examples = []
        self.test_examples = []
