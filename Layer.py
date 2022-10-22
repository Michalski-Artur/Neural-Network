from numpy import random
from typing import List

from Neuron import Neuron


class Layer:
    neurons = [Neuron]
    previous_layer_has_bias = None

    def __init__(self, neurons: List[Neuron], previous_layer_has_bias):
        self.neurons = neurons
        for (index, neuron) in enumerate(neurons):
            neuron.set_neuron_number_in_layer(index)
        self.previous_layer_has_bias = previous_layer_has_bias

    def get_layer_size(self):
        return len(self.neurons)

    def update_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)

    def set_weights(self, previous_layer_size: int):
        for neuron in self.neurons:
            neuron.set_weights((random.rand(previous_layer_size).tolist()))
