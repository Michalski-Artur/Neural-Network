from weights_handler import WeightsHandler

class Perceptron:
    def __init__(self, input_size: int, output_size: int, hidden_layers: int = 0, layer_size: int = 0, seed: int = None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.layer_size = layer_size
        self.seed = seed
        self.weights = WeightsHandler(input_size, output_size, hidden_layers, layer_size, seed)
