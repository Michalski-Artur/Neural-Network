import math
from activation_functions import ActivationFunctions
from data_visualizer import DataVisualizer
from data_manager import DataManager
from error_functions import ErrorFunctions
from network import Network
from network_enums import ProblemType

train_data_path = 'regression/data.activation.train.1000.csv'
test_data_path = 'regression/data.activation.test.1000.csv'
problem_type = ProblemType.REGRESSION
use_bias = True
hidden_layers_count = 1
hidden_layer_neurons_count = 1
activation_function = ActivationFunctions.identity
error_function = ErrorFunctions.mean_absolute_error
initial_seed = 1
learning_rate = 0.001
epoch_max = 1000
visualize_epochs = False

for hidden_layers_count in range(0,2):
    for error_function in [ErrorFunctions.mean_squared_error, ErrorFunctions.mean_absolute_error]:
        mse_sum = 0
        mae_sum = 0
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

            error_desc = 'mse' if error_function == ErrorFunctions.mean_squared_error else 'mae'
            mse = sum([ErrorFunctions.mean_squared_error(test_result[i], test_data['y'][i]) for i in range(len(test_result))]) / len(test_result)
            mae = sum([ErrorFunctions.mean_absolute_error(test_result[i], test_data['y'][i]) for i in range(len(test_result))]) / len(test_result)
            mse_sum += mse
            mae_sum += mae
            print(f'Mean squared error (for testing {error_desc} on {hidden_layers_count} layers): {mse:.3g}')
            print(f'Mean absolute error (for testing {error_desc} on {hidden_layers_count} layers): {mae:.3g}')
            DataVisualizer.visualize_data(problem_type, test_data, test_result, 'Test data', f'output/test_data_activation_layers_{hidden_layers_count}_{error_desc}_{seed}.png')
        print(f'Total mse (for testing {error_desc} on {hidden_layers_count} layers): {mse_sum/5:.3g}')
        print(f'Total mae (for testing {error_desc} on {hidden_layers_count} layers): {mae_sum/5:.3g}')

# Linear regression coefficients
if problem_type == ProblemType.REGRESSION:
    (a, b) = network.get_neuron_weights(-1, -1)
    print(f'Linear regression coefficients: a = {a:.3f}, b = {b:.3f}')
