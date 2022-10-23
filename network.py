import numpy as np
import pandas as pd
from graphviz import Digraph
from activation_functions import ActivationFunctionType, ActivationFunctions
from error_functions import ErrorFunctionType

from layer import Layer
from neuron import Neuron

class Network:
    def __init__(self, input_size: int, output_size: int, use_bias: bool, hidden_layers_count: int, hidden_layer_neurons_count: int,
                activation_function: ActivationFunctionType, error_function: ErrorFunctionType, initial_seed: int, learning_rate: float, epoch_max: int) -> None:

        self.__learning_rate = learning_rate
        self.__epoch_no = epoch_max
        self.__input_size = input_size
        self.__error_function = error_function
        self.__error_threshold = 1e-6
        np.random.seed(initial_seed)
        basic_neuron = Neuron(activation_function, error_function)
        self.__layers: list[Layer] = []
        hidden_layers = [Layer([basic_neuron.copy_neuron() for _ in range(hidden_layer_neurons_count)], use_bias) for _ in range(hidden_layers_count)]
        output_layer = Layer([basic_neuron.copy_neuron() for _ in range(output_size)], use_bias)
        for hidden_layer in hidden_layers:
            self.add_network_layer(hidden_layer)
        self.add_network_layer(output_layer)

    def add_network_layer(self, layer: Layer) -> None:
        previous_layer_size: int = self.__input_size if len(self.__layers) == 0 else len(self.__layers[-1].neurons)
        if layer.previous_layer_has_bias:
            previous_layer_size += 1
        self.__layers.append(layer)
        layer.set_weights(previous_layer_size)

    def train(self, training_set: pd.DataFrame, visualize: bool = False) -> None:
        current_iteration = 0
        mean_error = 1.0
        while not self.__stop_condition_met(current_iteration, mean_error):
            current_iteration += 1
            if current_iteration == self.__epoch_no:
                pass
            training_set = training_set.sample(frac=1)  # shuffle training set
            x_train, y_train = training_set.values[:, :-1], training_set.values[:, -1]
            mean_error = 0.0
            input_value: np.ndarray
            expected_result: float
            sample_len = len(x_train)
            for (input_value, expected_result) in zip(x_train, y_train):
                outputs = self.__forward_pass(input_value)
                expected = [0 for _ in range(len(outputs))]
                expected[int(expected_result) - 1] = 1
                mean_error += self.__error_function(outputs[int(expected_result) - 1], 1) / sample_len
                self.__backward_pass(expected)
                self.__update_weights()
            if visualize:
                self.visualize(current_iteration, current_iteration == self.__epoch_no)
            print(f'Training in progress (epoch={current_iteration}/{self.__epoch_no}). Mean error (classification)= {mean_error:.3g}')
        if not visualize:
            self.visualize(current_iteration, True)

    def get_classification_result(self, test_set: pd.DataFrame) -> list[int]:
        predictions = []
        x_test = test_set.values[:, :-1]
        input_value: np.ndarray
        for input_value in x_test:
            output = self.__forward_pass(input_value)
            prediction = output.index(max(output)) + 1
            predictions.append(prediction)
        return predictions

    def visualize(self, epoch_number: int, view: bool = False) -> None:
        graph = Digraph('G', filename=f'output/network_{epoch_number}', format='png')
        graph.attr('graph', pad='1', ranksep='5', nodesep='0.3', label=f'Network in epoch {epoch_number}', labelloc='t', fontsize='50')
        graph.attr('node', shape='circle')
        graph.attr('edge')
        for i in range(self.__input_size):
            graph.node(f'0_{i}', f'Input{i}')
        for i, layer in enumerate(self.__layers):
            for j, neuron in enumerate(layer.neurons):
                node_id = f'{i+1}_{j}'
                label = f'Output{j}' if i == len(self.__layers) - 1 else ''
                graph.node(node_id, label)
                for k, weight in enumerate(neuron.weights):
                    prev_node_id = f'{i}_{k}'
                    if layer.previous_layer_has_bias and k == len(neuron.weights) - 1:
                        graph.node(prev_node_id, f'Bias{i}')
                    graph.edge(prev_node_id, node_id, label=f'w={weight:.4g}\ne={neuron.delta:.4g}')
        graph.render(view=view)

    def __forward_pass(self, input_value: np.ndarray) -> list[float]:
        output = []
        for (layer_index, layer) in enumerate(self.__layers):
            if layer.previous_layer_has_bias:
                input_value = np.append(input_value, 1)
            output = layer.calculate_output(input_value, layer_index == len(self.__layers) - 1)
            input_value = output
        return output

    def __backward_pass(self, expected_result: list[float]) -> None:
        for (layer_index, layer) in reversed(list(enumerate(self.__layers))):
            next_layer = self.__layers[layer_index + 1] if layer_index + 1 < len(self.__layers) else None
            for neuron in layer.neurons:
                neuron.calculate_error(expected_result, next_layer)

    def __stop_condition_met(self, current_iteration: int, mean_error: float) -> bool:
        if current_iteration >= self.__epoch_no:
            print(f'Finished learning after reaching limit of {current_iteration} epochs')
            return True
        elif abs(mean_error) < self.__error_threshold:
            print(f'Early stopping at epoch {current_iteration} due to error {mean_error:.4g} < {self.__error_threshold}')
            return True
        return False

    def __update_weights(self) -> None:
        for layer in self.__layers:
            layer.update_weights(self.__learning_rate)
