import numpy as np
from activation_functions import ActivationFunctions
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
            neuron.set_weights((np.random.rand(previous_layer_size).tolist()))

    def calculate_output(self, input_value: list[float], is_output: bool) -> list[float]:
        if is_output:
            output = ActivationFunctions.softmax_vector([np.dot(neuron.get_weights(), input_value) for neuron in self.neurons])
            output = [neuron.calculate_output(input_value, output[neuron.neuron_index_in_layer]) for neuron in self.neurons]
        else:
            output = [neuron.calculate_output(input_value) for neuron in self.neurons]
        return output
