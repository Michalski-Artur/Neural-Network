from mnist import MNIST
import pandas as pd

from data_manager import DataManager
from network_enums import ProblemType


class MnistDataManager(DataManager):
    def __init__(self) -> None:
        super().__init__(train_data_path='', test_data_path='', problem_type=ProblemType.CLASSIFICATION)
        self.__mndata = MNIST('./mnist')

    def read_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = self.__mndata.load_training()
        test = self.__mndata.load_testing()
        self.training_data = pd.DataFrame(train[0])
        self.training_data = self.training_data / 255.0
        self.training_data['cls'] = train[1]
        self.training_data['cls'] = self.training_data['cls'] + 1
        self.testing_data = pd.DataFrame(test[0])
        self.testing_data = self.testing_data / 255.0
        self.testing_data['cls'] = test[1]
        self.testing_data['cls'] = self.testing_data['cls'] + 1

        self.training_data = self.training_data.sample(n=1000)
        self.testing_data = self.testing_data.sample(n=1000)
        return (self.training_data, self.testing_data)
