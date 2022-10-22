import matplotlib.pyplot as plt
from pandas import DataFrame


class DataVisualizer:
    @staticmethod
    def visualize_classification_data(df: DataFrame, output_file_name = None) -> None:
        first_class = df[(df['cls'] == 1)]
        first_class_correct = first_class[(first_class['cls'] == first_class['cls_trained'])]
        first_class_incorrect = first_class[(first_class['cls'] != first_class['cls_trained'])]
        second_class = df[(df['cls'] == 2)]
        second_class_correct = second_class[(second_class['cls'] == second_class['cls_trained'])]
        second_class_incorrect = second_class[(second_class['cls'] != second_class['cls_trained'])]

        ax = first_class_correct.plot.scatter(x='x', y='y', c='g', label='Correct first class', marker = 'o')
        second_class_correct.plot.scatter(x='x', y='y', ax=ax, c='g', label='Correct second class', marker = '*')
        first_class_incorrect.plot.scatter(x='x', y='y', ax=ax, c='r', label='Incorrect first class', marker = 'o')
        second_class_incorrect.plot.scatter(x='x', y='y', ax=ax, c='r', label='Incorrect second class', marker = '*')

        plt.title('Training data')
        if output_file_name is not None:
            plt.savefig(output_file_name)
        plt.show()