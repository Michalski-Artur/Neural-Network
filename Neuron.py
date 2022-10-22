import numpy as np
from numpy import dot


class Neuron:
    def __init__(self, activation_function, activation_function_derivative):
        self.input = None
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.weights = []
        self.output = 0
        self.error = 0
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

# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# https://towardsdatascience.com/coding-a-neural-network-from-scratch-in-numpy-31f04e4d605
# modify algorithm to use square error and its derivative: (a - y)**2 -> d_error = 2(a - y) instead of 1
    def calculate_error(self, expected_output, next_layer):
        error = 0
        if next_layer is None:
            error = self.output - expected_output
        else:
            for neuron in next_layer.neurons:
                error += neuron.weights[self.neuron_index_in_layer] * neuron.error
        self.error = error * self.activation_function_derivative(self.output)
        return self.error

    def update_weights(self, learning_rate):
        delta_weights = np.multiply(self.input, self.error * learning_rate)
        np.add(self.weights, -delta_weights)
