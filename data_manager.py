import pandas as pd

from network_enums import ProblemType

class DataManager:
    def __init__(self, train_data_path: str, test_data_path: str, problem_type: ProblemType) -> None:
        self.__train_data_path = train_data_path
        self.__test_data_path = test_data_path
        self.__problem_type = problem_type
        self.testing_data = None
        self.training_data = None

    def read_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.training_data = pd.read_csv(self.__train_data_path)
        self.testing_data = pd.read_csv(self.__test_data_path)
        if self.__problem_type is ProblemType.REGRESSION:
            column = 'y'
            min = self.training_data[column].min()
            max = self.training_data[column].max()
            self.training_data[column] = (self.training_data[column] - min)/(max - min)
            self.testing_data[column] = (self.testing_data[column] - self.testing_data[column].min())/(max - min)
        return (self.training_data, self.testing_data)

    def get_input_size(self) -> int:
        return len(self.training_data.columns) - 1

    def get_output_layer_size(self) -> int:
        if self.__problem_type == ProblemType.CLASSIFICATION:
            return self.training_data.iloc[:, -1].max()
        elif self.__problem_type == ProblemType.REGRESSION:
            return 1
