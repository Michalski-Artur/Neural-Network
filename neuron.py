import numpy as np
from numpy import dot, ndarray

from activation_functions import ActivationFunctionType
from error_functions import ErrorFunctionType


class Neuron:
    def __init__(self, activation_function: ActivationFunctionType, error_function: ErrorFunctionType) -> None:
        self.weights = []
        self.delta = 0
        self.neuron_index_in_layer = 0
        self.__activation_function = activation_function
        self.__error_function = error_function
        self.__input = None
        self.__output = 0

    def set_neuron_number_in_layer(self, index: int) -> None:
        self.neuron_index_in_layer = index

    def copy_neuron(self) -> 'Neuron':
        return Neuron(self.__activation_function, self.__error_function)

    def set_weights(self, weights: list[float]) -> None:
        self.weights = weights

    def get_weights(self) -> list[float]:
        return self.weights

    def calculate_output(self, input_value: ndarray) -> float:
        self.__input = input_value
        self.__output = self.__activation_function(dot(self.weights, input_value))
        return self.__output

    # Based on: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    def calculate_error(self, expected_output: list[float], next_layer) -> float:
        # TODO: Find out what's going on here
        error = 0
        if next_layer is None:
            error = self.__error_function(self.__output, expected_output[self.neuron_index_in_layer], True)
        else:
            for neuron in next_layer.neurons:
                error += neuron.weights[self.neuron_index_in_layer] * neuron.delta
        self.delta = error * self.__activation_function(self.__output, True)
        return self.delta

    def update_weights(self, learning_rate: float) -> None:
        delta_weights = np.multiply(self.__input, self.delta * learning_rate)
        self.weights = np.add(self.weights, -delta_weights)
