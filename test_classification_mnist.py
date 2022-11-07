from activation_functions import ActivationFunctions
from data_visualizer import DataVisualizer
from error_functions import ErrorFunctions
from mnist_data_manager import MnistDataManager
from network import Network
from network_enums import ProblemType


train_data_path = 'classification/data.noisyXOR.train.1000.csv'
test_data_path = 'classification/data.noisyXOR.test.1000.csv'
problem_type = ProblemType.CLASSIFICATION
use_bias = True
hidden_layers_count = 2
hidden_layer_neurons_count = 10
activation_function = ActivationFunctions.relu
error_function = ErrorFunctions.mean_squared_error
initial_seed = 1
learning_rate = 0.001
epoch_max = 100
visualize_epochs = False


 # Read data
data_manager = MnistDataManager()
data_manager.read_data()
training_data = data_manager.training_data
input_size = data_manager.get_input_size()
output_layer_size = data_manager.get_output_layer_size()
# hidden_layer_neurons_count = math.isqrt(input_size * output_layer_size)


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
network.train(data_manager.training_data, visualize_epochs)


test_data = data_manager.testing_data
test_result = network.predict(test_data)
accuracy = sum([1 if test_data['cls'].values[i] == test_result[i] else 0 for i in range(len(test_data))]) / len(test_data)
print(f'Accuracy: {accuracy*100:.3f}%')
DataVisualizer.print_mnist_result(test_data, test_result)
