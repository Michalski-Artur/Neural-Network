from activation_functions import ActivationFunctions
from data_visualizer import DataVisualizer
from error_functions import ErrorFunctions
from mnist_data_manager import MnistDataManager
from network import Network
from network_enums import ProblemType

# Numbers are stored in images table as 28*28 arrays. Each element of the array ranges from 0 to 255 indicating grey scale
# Labels store number presented in i-th image

problem_type = ProblemType.CLASSIFICATION
use_bias = True
hidden_layers_count = 1
hidden_layer_neurons_count = 100
activation_function = ActivationFunctions.sigmoid
error_function = ErrorFunctions.mean_squared_error
initial_seed = 256
learning_rate = 0.1
epoch_max = 100
visualize_epochs = False

print('Reading MNIST data...')
data_manager = MnistDataManager(initial_seed)
data_manager.read_data()
input_size = data_manager.get_input_size()
output_layer_size = data_manager.get_output_layer_size()
print('MNIST Data read.')

# Create network structure and train it
network = Network(
    problem_type=problem_type,
    input_size=input_size,
    output_size=output_layer_size,
    use_bias=use_bias,
    hidden_layers_count=hidden_layers_count,
    hidden_layer_neurons_count=hidden_layer_neurons_count,
    activation_function=activation_function,
    error_function=error_function,
    initial_seed=initial_seed,
    learning_rate=learning_rate,
    epoch_max=epoch_max)

network.train(data_manager.training_data, False)

testing_result = network.predict(data_manager.testing_data)
DataVisualizer.visualize_mnist_data(data_manager.testing_data, testing_result)
