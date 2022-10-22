from ActivationFunctions import ActivationFunctions
from DataVisualizer import DataVisualizer
from DataManager import DataManager
from Layer import Layer
from Network import Network
from Neuron import Neuron

network = Network(
    learning_rate=0.1,
    epoch_no=10,
    initial_seed=1,
    input_size=2
)

basic_neuron = Neuron(
    activation_function=ActivationFunctions.sigmoid,
    activation_function_derivative=ActivationFunctions.sigmoid_derivative
)

hidden_layer_neurons_count = 30
hidden_layer = Layer(
    neurons=[basic_neuron.copy_neuron() for _ in range(hidden_layer_neurons_count)],
    previous_layer_has_bias=False
)

output_layer_neurons_count = 1
output_layer = Layer(
    neurons=[basic_neuron.copy_neuron() for _ in range(output_layer_neurons_count)],
    previous_layer_has_bias=False
)

network.add_network_layer(hidden_layer)
network.add_network_layer(output_layer)

data_manager = DataManager("classification/data.simple.test.100.csv", True)

df = data_manager.get_data()
DataVisualizer.visualize_data(df)

network.learn(data_manager.get_training_data())
output = network.compute(data_manager.get_test_data())
