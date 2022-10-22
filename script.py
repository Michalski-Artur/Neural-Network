from ActivationFunctions import ActivationFunctions
from DataVisualizer import DataVisualizer
from DataManager import DataManager
from Layer import Layer
from Network import Network
from Neuron import Neuron

# Training data loading
data_manager = DataManager("classification/data.simple.train.100.csv")
df = data_manager.get_data()
# DataVisualizer.visualize_data(df)

# Network configuration
input_size = len(df.values[0]) - 1
network = Network(
    learning_rate=0.1,
    epoch_no=50,
    initial_seed=1,
    input_size=input_size
)

basic_neuron = Neuron(
    activation_function=ActivationFunctions.sigmoid,
    activation_function_derivative=ActivationFunctions.sigmoid_derivative
)

# Layers configuration
hidden_layer_neurons_count = 2
hidden_layer = Layer(
    neurons=[basic_neuron.copy_neuron() for _ in range(hidden_layer_neurons_count)],
    previous_layer_has_bias=True
)

classes_count = len(set([row[-1] for row in df.values]))
output_layer_neurons_count = classes_count  # Should match number of classes -> one hot encoding
output_layer = Layer(
    neurons=[basic_neuron.copy_neuron() for _ in range(output_layer_neurons_count)],
    previous_layer_has_bias=True
)

network.add_network_layer(hidden_layer)
network.add_network_layer(output_layer)

# Training of network
network.train(df)

# Prediction on data from test set
data_manager = DataManager("classification/data.simple.test.100.csv")
df = data_manager.get_data()
output = network.predict(df)
