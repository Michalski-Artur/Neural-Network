from mnist import MNIST


class MnistDataManager:
    def __init__(self) -> None:
        self.mndata = MNIST('./mnist')

    def read_data(self, is_test_data: bool = False):
        (images, labels) = self.mndata.load_testing() if is_test_data else self.mndata.load_training()
        new_images = []
        for (index, image) in enumerate(images):
            new_image = [x / 255.0 for x in image]
            new_image.append(labels[index])
            new_images.append(new_image)
            if index > 10000:
                break
        return new_images
    def get_input_size(self) -> int:
        return 28*28

    def get_output_layer_size(self) -> int:
        return 10
