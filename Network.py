import numpy as np
import pandas as pd
from graphviz import Digraph

from Layer import Layer
from Neuron import Neuron

class Network:
    def __init__(self, input_size: int, output_size: int, use_bias: bool, hidden_layers_count: int, hidden_layer_neurons_count: int, activation_function: callable, initial_seed: int, learning_rate: float, epoch_max: int) -> None:
        self.learning_rate = learning_rate
        self.epoch_no = epoch_max
        self.input_size = input_size
        np.random.seed(initial_seed)
        basic_neuron = Neuron(activation_function)
        self.layers: list[Layer] = []
        hidden_layers = [Layer([basic_neuron.copy_neuron() for _ in range(hidden_layer_neurons_count)], use_bias) for _ in range(hidden_layers_count)]
        output_layer = Layer([basic_neuron.copy_neuron() for _ in range(output_size)], use_bias)
        for hidden_layer in hidden_layers:
            self.add_network_layer(hidden_layer)
        self.add_network_layer(output_layer)

    def add_network_layer(self, layer: Layer) -> None:
        previous_layer_size: int = self.input_size if len(self.layers) == 0 else len(self.layers[-1].neurons)
        if layer.previous_layer_has_bias:
            previous_layer_size += 1
        self.layers.append(layer)
        layer.set_weights(previous_layer_size)

    def forward_pass(self, input_value: np.ndarray) -> list[float]:
        output = []
        for layer in self.layers:
            if layer.previous_layer_has_bias:
                input_value = np.append(input_value, 1)
            output = []
            for neuron in layer.neurons:
                output.append(neuron.calculate_output(input_value))
            input_value = output
        return output

    def backward_pass(self, expected_result: float) -> None:
        for (layer_index, layer) in reversed(list(enumerate(self.layers))):
            next_layer = self.layers[layer_index + 1] if layer_index + 1 < len(self.layers) else None
            for neuron in layer.neurons:
                neuron.calculate_error(expected_result, next_layer)

    def learn(self, training_set: pd.DataFrame, visualize: bool = False) -> None:
        current_iteration = 0
        while not self.stop_condition_met(current_iteration):
            current_iteration += 1
            print(f'Epoch: {current_iteration}')
            training_set = training_set.sample(frac=1)  # shuffle training set
            x_train, y_train = training_set.values[:, :-1], training_set.values[:,-1]
            input_value: np.ndarray
            expected_result: float
            for (input_value, expected_result) in zip(x_train, y_train):
                self.forward_pass(input_value)
                self.backward_pass(expected_result)
                self.update_weights()
            if visualize:
                self.visualize_network(current_iteration)
        print(f'Finished learning after {current_iteration} epochs')

    def stop_condition_met(self, current_iteration: int) -> bool:
        # TODO: Maybe more conditions?
        return current_iteration >= self.epoch_no

    def compute(self, test_set: pd.DataFrame) -> list[list[float]]:
        output = []
        x_test = test_set.values[:, :-1]
        input_value: np.ndarray
        for input_value in x_test:
            output.append(self.forward_pass(input_value))
        return output

    def update_weights(self) -> None:
        for layer in self.layers:
            layer.update_weights(self.learning_rate)

    def visualize_network(self, epoch_number: int) -> None:
        graph = Digraph('G', filename=f'output/network_{epoch_number}', format='png')
        graph.attr('graph', pad='1', ranksep='5', nodesep='0.3', label=f'Network in epoch {epoch_number}', labelloc='t', fontsize='50')
        graph.attr('node', shape='circle')
        graph.attr('edge')
        for i in range(self.input_size):
            graph.node(f'0_{i}', f'Input_{i}')
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                node_id = f'{i+1}_{j}'
                label = f'Output_{j}' if i == len(self.layers) - 1 else ''
                graph.node(node_id, label)
                for k, weight in enumerate(neuron.weights):
                    graph.edge(f'{i}_{k}', node_id, label=f'w={weight:.4f}\ne={neuron.error:.4f}')
        graph.render(view=False)
