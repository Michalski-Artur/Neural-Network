import numpy as np
from numpy import dot


class Neuron:
    def __init__(self, activation_function, activation_function_derivative):
        self.input = None
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.weights = []
        self.output = 0
        self.delta = 0
        self.neuron_index_in_layer = 0

    def set_neuron_number_in_layer(self, index):
        self.neuron_index_in_layer = index

    def copy_neuron(self):
        return Neuron(self.activation_function, self.activation_function_derivative)

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def calculate_output(self, input):
        self.input = input
        self.output = self.activation_function(dot(self.weights, input))
        return self.output

# Based on: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    def calculate_error(self, expected_output, next_layer):
        error = 0
        if next_layer is None:
            error = self.output - expected_output[self.neuron_index_in_layer]
        else:
            for neuron in next_layer.neurons:
                error += neuron.weights[self.neuron_index_in_layer] * neuron.delta
        self.delta = error * self.activation_function_derivative(self.output)
        return self.delta

    def update_weights(self, learning_rate):
        delta_weights = np.multiply(self.input, self.delta * learning_rate)
        self.weights = np.add(self.weights, -delta_weights)
