import math
from activation_functions import ActivationFunctions
from data_visualizer import DataVisualizer
from data_manager import DataManager
from error_functions import ErrorFunctions
from network import Network
from network_enums import ProblemType

train_data_path = 'classification/data.three_gauss.train.1000.csv'
test_data_path = 'classification/data.three_gauss.test.1000.csv'
problem_type = ProblemType.CLASSIFICATION
use_bias = True
hidden_layers_count = 1
hidden_layer_neurons_count = 1
activation_function = ActivationFunctions.relu
error_function = ErrorFunctions.mean_squared_error
initial_seed = 1
learning_rate = 0.001
epoch_max = 100
visualize_epochs = False

for hidden_layers_count in range(5):
    accuracy = 0
    for seed in [1, 165, 2521, 12345, 842107]:
    # Read data
        data_manager = DataManager(train_data_path, test_data_path, problem_type)
        data_manager.read_data()
        training_data = data_manager.training_data
        input_size = data_manager.get_input_size()
        output_layer_size = data_manager.get_output_layer_size()
        hidden_layer_neurons_count = math.isqrt(input_size * output_layer_size)

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
            initial_seed=seed,
            learning_rate=learning_rate,
            epoch_max=epoch_max)
        network.train(data_manager.training_data, visualize_epochs)

        training_result = network.predict(training_data)
        DataVisualizer.visualize_data(problem_type, training_data, training_result, 'Training data', 'output/training_data.png')

        test_data = data_manager.testing_data
        test_result = network.predict(test_data)
        accuracy += sum([1 if test_data['cls'][i] == test_result[i] else 0 for i in range(len(test_data))]) / len(test_data)
        DataVisualizer.visualize_data(problem_type, test_data, test_result, 'Test data', f'output/test_layers_{hidden_layers_count}_relu_{seed}.png')
    print(f'Total accuracy ({hidden_layers_count} layers): {accuracy/5*100:.3f}%')
