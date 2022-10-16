import pandas as pd


class DataManager:

    def __init__(self, filename, training_frac=0.8, random_state=10):
        self.training_frac = training_frac
        self.random_state = random_state
        self.testing_data = None
        self.training_data = None
        self.data = None
        self.read_file(filename)

    def read_file(self, filename):
        self.data = pd.read_csv(filename)
        self.training_data = self.data.sample(frac=self.training_frac, random_state=self.random_state)
        self.testing_data = self.data.drop(self.training_data.index)
        return self.data

    def get_data(self):
        return self.data

    def get_training_data(self):
        return self.training_data

    def get_test_data(self):
        return self.testing_data
