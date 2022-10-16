from numpy import random


class Network:

    def __init__(self, learning_rate, epoch_no, initial_seed, input_size):
        self.learning_rate = learning_rate
        self.epoch_no = epoch_no
        self.input_size = input_size
        self.layers = []
        random.seed(initial_seed)

    def add_network_layer(self, layer):
        previous_layer_size = self.input_size if len(self.layers) == 0 else self.layers[-1]
        if layer.previous_layer_has_bias:
            previous_layer_size += 1

        self.layers.append(layer)
        for neuron in layer.neurons:
            neuron.set_weights(random.rand(previous_layer_size))

    def forward_pass(self, input):
        output = []
        for layer in self.layers:
            if layer.previous_layer_has_bias:
                input.append(1)
            output = []
            for neuron in layer.neurons:
                output.append(neuron.calculate_output(input))
            input = output
        return output

    def backward_pass(self, result, expected_result):
        raise NotImplementedError

    def learn(self, training_set):
        self.epoch_no = 0
        current_iteration = 0
        while not self.stop_condition_met(current_iteration):
            current_iteration += 1
            delta_weights = 0
            training_set = training_set.sample(frac=1)  # shuffle training set
            sample = [(row[:-1], row[-1]) for row in training_set.values]
            for row in training_set.itertuples(index=False):
                input = row[:-1]
                expected_result = row[-1]
                result = self.forward_pass(input)
                delta_weights = self.backward_pass(result, expected_result)

    def stop_condition_met(self, current_iteration):
        return current_iteration > self.epoch_no
