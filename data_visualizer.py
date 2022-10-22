import matplotlib.pyplot as plt
from pandas import DataFrame

class DataVisualizer:
    @staticmethod
    def visualize_classification_data(data: DataFrame, result: list[int], title: str, output_file_name = None) -> None:
        data['cls_calculated'] = result
        first_class = data[(data['cls'] == 1)]
        first_class_correct = first_class[(first_class['cls'] == first_class['cls_calculated'])]
        first_class_incorrect = first_class[(first_class['cls'] != first_class['cls_calculated'])]
        second_class = data[(data['cls'] == 2)]
        second_class_correct = second_class[(second_class['cls'] == second_class['cls_calculated'])]
        second_class_incorrect = second_class[(second_class['cls'] != second_class['cls_calculated'])]

        axes = first_class_correct.plot.scatter(x='x', y='y', c='g', label=f'Correct first class ({len(first_class_correct)/len(first_class)*100:.3f}%)', marker = 'x')
        first_class_incorrect.plot.scatter(x='x', y='y', ax=axes, c='r', label=f'Incorrect first class ({len(first_class_incorrect)/len(first_class)*100:.3f}%)', marker = 'x')
        second_class_correct.plot.scatter(x='x', y='y', ax=axes, c='g', label=f'Correct second class ({len(second_class_correct)/len(second_class)*100:.3f}%)', marker = 's')
        second_class_incorrect.plot.scatter(x='x', y='y', ax=axes, c='r', label=f'Incorrect second class ({len(second_class_incorrect)/len(second_class)*100:.3f}%)', marker = 's')

        plt.title(title)
        if output_file_name is not None:
            plt.savefig(output_file_name)
        plt.show()