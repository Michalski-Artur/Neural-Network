import matplotlib.pyplot as plt
from pandas import DataFrame


class DataVisualizer:

    @staticmethod
    def visualize_data(df: DataFrame):
        first_class = df[(df['cls'] == 1)]
        ax = first_class.plot.scatter(x='x', y='y', c='g', label='First class')
        second_class = df[(df['cls'] == 2)]
        second_class.plot.scatter(x='x', y='y', ax=ax, c='r', label='Second class')
        plt.title('Input data')
        plt.show()
