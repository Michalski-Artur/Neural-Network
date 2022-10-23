import math
import sys

from activation_functions import ActivationFunctionType, ActivationFunctions
from data_manager import DataManager
from data_visualizer import DataVisualizer
from error_functions import ErrorFunctionType, ErrorFunctions
from network import Network
from network_enums import ProblemType

LEARNING_RATE = 0.1
EPOCH_MAX = 1
INTIIAL_SEED = 1
HIDDEN_LAYERS_COUNT = 2


def __main__() -> None:
    # Input arguments - train_data_path, test_data_path, problem_type (c for classification, r for regression), visualize_epochs, number_of_layers, number_of_neurons_per_layer, use_bias, activation_function, error_function, initial_seed
    # Example: python script.py "classification/data.simple.test.100.csv" "classification/data.simple.test.100.csv" c 1 2 30 1 sigmoid cross_entropy 1

    # Raise exception if the number of arguments is not correct
    if len(sys.argv) < 4:
        raise Exception("Not enough arguments")

    # Parse input arguments
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    problem_type = __parse_problem_type(sys.argv[3])
    # Below are the arguments that should have default values
    visualize_epochs = sys.argv[4] = bool(sys.argv[4]) or False
    hidden_layers_count = int(sys.argv[5]) or HIDDEN_LAYERS_COUNT
    use_bias = bool(sys.argv[7]) or True
    activation_function = __parse_activation_function(sys.argv[8])
    error_function = __parse_error_function(sys.argv[9])
    initial_seed = int(sys.argv[10]) or INTIIAL_SEED

    # Read data
    data_manager = DataManager(train_data_path, test_data_path, problem_type)
    data_manager.read_data()
    input_size = data_manager.get_input_size()
    output_layer_size = data_manager.get_output_layer_size()
    hidden_layer_neurons_count = int(sys.argv[6]) or math.isqrt(input_size * output_layer_size)

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
        learning_rate=LEARNING_RATE,
        epoch_max=EPOCH_MAX)

    training_data = data_manager.training_data
    network.train(training_data, visualize_epochs)
    training_result = network.predict(training_data)
    DataVisualizer.visualize_classification_data(training_data, training_result, 'Training data', 'output/training_data.png')

    test_data = data_manager.testing_data
    test_result = network.predict(test_data)
    DataVisualizer.visualize_classification_data(test_data, test_result, 'Test data', 'output/test_data.png')

def __parse_problem_type(problem_type: str) -> any:
    if problem_type == 'c':
        return ProblemType.CLASSIFICATION
    elif problem_type == 'r':
        return ProblemType.REGRESSION
    else:
        raise ValueError('Supplied problem type not supported (use \'c\' for classification or \'r\' for regression)')

def __parse_activation_function(activation_function: str) -> ActivationFunctionType:
    if activation_function == 'sigmoid':
        return ActivationFunctions.sigmoid
    elif activation_function == 'tanh':
        return ActivationFunctions.tanh
    else:
        print('Activation function not supported, using sigmoid')
        return ActivationFunctions.sigmoid

def __parse_error_function(error_function: str) -> ErrorFunctionType:
    if error_function == 'cross_entropy':
        return ErrorFunctions.cross_entropy_error
    elif error_function == 'mean_squared':
        return ErrorFunctions.mean_squared_error
    elif error_function == 'mean_absolute':
        return ErrorFunctions.mean_absolute_error
    else:
        print('Error function not supported, using cross_entropy')
        return ErrorFunctions.cross_entropy_error

if __name__ == "__main__":
    __main__()
