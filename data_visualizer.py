import matplotlib.pyplot as plt
from pandas import DataFrame

from network_enums import ProblemType

class DataVisualizer:
    __markers = ['x', 's', '^', 'd', 'X', 'o']

    @staticmethod
    def visualize_data(problem_type: ProblemType, data: DataFrame, result: list[float], title: str, output_file_name = None) -> None:
        if problem_type == ProblemType.CLASSIFICATION:
            DataVisualizer.__visualize_classification_data(data, result, title, output_file_name)
        elif problem_type == ProblemType.REGRESSION:
            DataVisualizer.__visualize_regression_data(data, result, title, output_file_name)

    @staticmethod
    def __visualize_classification_data(data: DataFrame, result: list[int], title: str, output_file_name = None) -> None:
        data['cls_calculated'] = result
        number_of_classes = data['cls'].max()
        correct = []
        incorrect = []
        for i in range(number_of_classes):
            class_data = data[data['cls'] == i + 1]
            correct.append(class_data[class_data['cls'] == class_data['cls_calculated']])
            incorrect.append(class_data[class_data['cls'] != class_data['cls_calculated']])
        axes = None
        for i in range(number_of_classes):
            correctness = len(correct[i]) / (len(correct[i]) + len(incorrect[i]))
            axes = correct[i].plot.scatter(x='x', y='y', c='green', ax=axes, label=f'Correct class {i+1} ({correctness * 100:.3f}%)', marker = DataVisualizer.__markers[i % len(DataVisualizer.__markers)])
            axes = incorrect[i].plot.scatter(x='x', y='y', c='red', ax=axes, label=f'Incorrect class {i+1} ({(1 - correctness) * 100:.3f}%)', marker = DataVisualizer.__markers[i % len(DataVisualizer.__markers)])

        plt.title(title)
        if output_file_name is not None:
            plt.savefig(output_file_name)
        plt.show()

    @staticmethod
    def __visualize_regression_data(data: DataFrame, result: list[float], title: str, output_file_name = None) -> None:
        data['y_calculated'] = result
        axes = data.plot.scatter(x='x', y='y', c='b', label='Expected results', marker = 'x')
        data.plot.scatter(x='x', y='y_calculated', ax=axes, c='gold', label='Regression results', marker = '*')
        plt.title(title)
        if output_file_name is not None:
            plt.savefig(output_file_name)
        plt.show()
