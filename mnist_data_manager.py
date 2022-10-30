from mnist import MNIST


class MnistDataManager:
    def __init__(self) -> None:
        self.mndata = MNIST('./mnist')

    def read_data(self, is_test_data: bool = False):
        return self.mndata.load_testing() if is_test_data else self.mndata.load_training()

    def get_input_size(self) -> int:
        return 28*28

    def get_output_layer_size(self) -> int:
        return 10
