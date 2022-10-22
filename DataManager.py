import pandas as pd


class DataManager:

    def __init__(self, filename):
        self.data = None
        self.read_file(filename)

    def read_file(self, filename):
        self.data = pd.read_csv(filename)
        return self.data

    def get_data(self):
        return self.data
