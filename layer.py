from numpy import random
from neuron import Neuron


class Layer:
    def __init__(self, neurons: list[Neuron], previous_layer_has_bias: bool) -> None:
        self.neurons = neurons
        self.previous_layer_has_bias = previous_layer_has_bias
        for (index, neuron) in enumerate(neurons):
            neuron.set_neuron_number_in_layer(index)

    def get_layer_size(self) -> int:
        return len(self.neurons)

    def update_weights(self, learning_rate: float) -> None:
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)

    def set_weights(self, previous_layer_size: int) -> None:
        for neuron in self.neurons:
            neuron.set_weights((random.rand(previous_layer_size).tolist()))
