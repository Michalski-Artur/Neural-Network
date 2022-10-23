from activation_functions import ActivationFunctions
from data_visualizer import DataVisualizer
from data_manager import DataManager
from network import Network
from network_enums import ProblemType

train_data_path = 'regression/data.activation.train.100.csv'
test_data_path = 'regression/data.activation.test.100.csv'
problem_type = ProblemType.REGRESSION
use_bias = True
hidden_layers_count = 0
hidden_layer_neurons_count = 1
activation_function = ActivationFunctions.identity
initial_seed = 1
learning_rate = 0.01
epoch_max = 5000
visualize_epochs = False

 # Read data
data_manager = DataManager(train_data_path, test_data_path, problem_type)
data_manager.read_data()
training_data = data_manager.training_data
input_size = data_manager.get_input_size()
output_layer_size = data_manager.get_output_layer_size()

# Create network structure and train it
network = Network(problem_type, input_size, output_layer_size, use_bias, hidden_layers_count, hidden_layer_neurons_count, activation_function, initial_seed, learning_rate, epoch_max)
network.train(data_manager.training_data, visualize_epochs)

training_result = network.predict(training_data)
# DataVisualizer.visualize_classification_data(training_data, training_result, 'Training data', 'output/training_data.png')

test_data = data_manager.testing_data
test_result = network.predict(test_data)
# DataVisualizer.visualize_classification_data(test_data, test_result, 'Test data', 'output/test_data.png')

# Linear regression coefficients
(a, b) = network.get_neuron_weights(-1, -1)
print(f'a={a}, b={b}')

