import numpy as np
from numpy import dot, ndarray


class Neuron:
    def __init__(self, activation_function: callable) -> None:
        self.activation_function = activation_function
        self.input = None
        self.weights = []
        self.output = 0
        self.error = 0
        self.neuron_index_in_layer = 0

    def set_neuron_number_in_layer(self, index: int) -> None:
        self.neuron_index_in_layer = index

    def copy_neuron(self) -> 'Neuron':
        return Neuron(self.activation_function)

    def set_weights(self, weights) -> None:
        self.weights = weights

    def get_weights(self) -> list[float]:
        return self.weights

    def calculate_output(self, input_value: ndarray) -> float:
        self.input = input_value
        self.output = self.activation_function(dot(self.weights, input_value))
        return self.output

    # https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    # https://towardsdatascience.com/coding-a-neural-network-from-scratch-in-numpy-31f04e4d605
    # modify algorithm to use square error and its derivative: (a - y)**2 -> d_error = 2(a - y) instead of 1
    def calculate_error(self, expected_output: float, next_layer) -> float:
        error = 0
        if next_layer is None:
            error = self.output - expected_output
        else:
            for neuron in next_layer.neurons:
                error += neuron.weights[self.neuron_index_in_layer] * neuron.error
        self.error = error * self.activation_function(self.output, True)
        return self.error

    def update_weights(self, learning_rate: float) -> None:
        delta_weights = np.multiply(self.input, self.error * learning_rate)
        np.add(self.weights, -delta_weights)
