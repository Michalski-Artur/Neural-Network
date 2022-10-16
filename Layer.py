from typing import List

from Neuron import Neuron


class Layer:
    def __init__(self, neurons: List[Neuron], previous_layer_has_bias):
        self.neurons = neurons
        self.previous_layer_has_bias = previous_layer_has_bias

    def get_layer_size(self):
        return len(self.neurons)
