import numpy as np
from numpy import dot, ndarray


class Neuron:
    def __init__(self, activation_function: callable) -> None:
        self.activation_function = activation_function
        self.input = None
        self.weights = []
        self.output = 0
        self.delta = 0
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

    # Based on: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    def calculate_error(self, expected_output: list[float], next_layer) -> float:
        error = 0
        if next_layer is None:
            error = self.output - expected_output[self.neuron_index_in_layer]
        else:
            for neuron in next_layer.neurons:
                error += neuron.weights[self.neuron_index_in_layer] * neuron.delta
        self.delta = error * self.activation_function(self.output, True)
        return self.delta

    def update_weights(self, learning_rate: float) -> None:
        delta_weights = np.multiply(self.input, self.delta * learning_rate)
        self.weights = np.add(self.weights, -delta_weights)
