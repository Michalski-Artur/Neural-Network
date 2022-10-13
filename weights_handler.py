import numpy as np

class WeightsHandler:
    def __init__(self, input_size: int, output_size: int, hidden_layers: int = 0, layer_size: int = 0, seed: int = None):
        self.__weights = []
        prev_size = input_size
        next_size = layer_size if hidden_layers > 0 else output_size
        np.random.seed(seed)
        for i in range(hidden_layers + 1):
            self.__weights.append(np.random.random_sample(size = (prev_size, next_size)))
            prev_size = next_size
            next_size = output_size if i == hidden_layers - 1 else layer_size

    def get_weights_in_layer(self, layer_number: int) -> np.ndarray:
        return self.__weights[layer_number]
