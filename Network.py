import numpy as np
import pandas as pd

from Layer import Layer


class Network:

    learning_rate = 0
    layers = []

    def __init__(self, learning_rate, epoch_no, initial_seed, input_size):
        self.learning_rate = learning_rate
        self.epoch_no = epoch_no
        self.input_size = input_size
        np.random.seed(initial_seed)

    def add_network_layer(self, layer: Layer):
        previous_layer_size: int = self.input_size if len(self.layers) == 0 else len(self.layers[-1].neurons)
        if layer.previous_layer_has_bias:
            previous_layer_size += 1

        self.layers.append(layer)
        layer.set_weights(previous_layer_size)

    def forward_pass(self, input: np.ndarray):
        output = []
        for layer in self.layers:
            if layer.previous_layer_has_bias:
                input = np.append(input, 1)
            output = []
            for neuron in layer.neurons:
                output.append(neuron.calculate_output(input))
            input = output
        return output

    def backward_pass(self, expected_result):
        for (layer_index, layer) in reversed(list(enumerate(self.layers))):
            next_layer = self.layers[layer_index + 1] if layer_index + 1 < len(self.layers) else None
            for neuron in layer.neurons:
                neuron.calculate_error(expected_result, next_layer)

    def learn(self, training_set: pd.DataFrame):
        current_iteration = 0
        while not self.stop_condition_met(current_iteration):
            current_iteration += 1
            training_set = training_set.sample(frac=1)  # shuffle training set
            x_train, y_train = training_set.values[:, :-1], training_set.values[:,-1]
            for (input, expected_result) in zip(x_train, y_train):
                self.forward_pass(input)
                self.backward_pass(expected_result)
                self.update_weights()

    def stop_condition_met(self, current_iteration):
        return current_iteration > self.epoch_no

    def compute(self, test_set: pd.DataFrame):
        output = []
        x_test = test_set.values[:, :-1]
        for input in x_test:
            output.append(self.forward_pass(input))
        return output

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights(self.learning_rate)
