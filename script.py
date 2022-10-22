from ActivationFunctions import ActivationFunctions
from DataVisualizer import DataVisualizer
from DataManager import DataManager
from Network import Network
from network_enums import ProblemType

train_data_path = 'classification/data.simple.train.100.csv'
test_data_path = 'classification/data.simple.test.100.csv'
problem_type = ProblemType.CLASSIFICATION
use_bias = False
hidden_layers_count = 1
hidden_layer_neurons_count = 2
activation_function = ActivationFunctions.sigmoid
initial_seed = 1
learning_rate = 0.1
epoch_max = 50

 # Read data
data_manager = DataManager(train_data_path, test_data_path, problem_type)
data_manager.read_data()
training_data = data_manager.training_data
input_size = data_manager.get_input_size()
output_layer_size = data_manager.get_output_layer_size()

# Create network structure and train it
network = Network(input_size, output_layer_size, use_bias, hidden_layers_count, hidden_layer_neurons_count, activation_function, initial_seed, learning_rate, epoch_max)
network.train(data_manager.training_data)

training_result = network.get_classification(training_data)
DataVisualizer.visualize_classification_data(training_data, training_result, 'Training data', 'output/training_data.png')

test_data = data_manager.testing_data
test_result = network.get_classification(test_data)
DataVisualizer.visualize_classification_data(test_data, test_result, 'Test data', 'output/test_data.png')
