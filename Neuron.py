class Neuron:
    def __init__(self, activation_function, activation_function_derivative):
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.weights = []
        self.output = 0

    def copy_neuron(self):
        return Neuron(self.activation_function, self.activation_function_derivative)

    def set_weights(self, weights):
        self.weights = weights

    def calculate_output(self, input):
        self.output = self.activation_function(self.weights * input)
        return self.output

    def improve_weights(self, delta_weights):
        self.weights += delta_weights
