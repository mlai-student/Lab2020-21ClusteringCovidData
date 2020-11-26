# TODO: Extract datageneration and move to data_generating
class Examples:
    def __init__(self, df=None, date=None, search_value=[], group=None):
        self.train_examples = []
        self.test_examples = []
        self.date, self.group, self.search_value, self.df = date, group, search_value, df
        if df is not None:
            self.groups = self.df[[date, group, search_val]].groupby(group)


    def make_examples(self, no_snippets=100, length=7, label_length=1, overlap=True, test_share=.1):
        examples = []
        # extract Examples from data
        for group in self.groups:
            group_sort = group[1].sort_values(by=self.date, ascending=True)
            max_idx = group_sort.shape[0]-length-label_length-1
            indices = make_interval_indices(overlap, length, no_snippets, max_idx)
            for [start, end] in indices:
                X = group_sort.iloc[start : end]
                Y = group_sort.iloc[end+1 : end+1+label_length]
                examples.append(Snippet(np.array(X[self.search_value]), np.array(Y[self.search_value])))

        # Divide examples into trainings and test examples
        test_share = round(test_share*len(examples))
        self.test_examples = random.sample(examples, test_share)
        self.train_examples = [x for x in examples if x not in self.test_examples]
        print("You created {} train examples and {} test examples".format(len(self.train_examples), len(self.test_examples)))


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


def make_interval_indices(length, no_intervals, max_idx):
    start_idxs = [random.randint(0, max_idx) for _ in range(no_intervals)]
    return [[x, x+length] for x in start_idxs]
