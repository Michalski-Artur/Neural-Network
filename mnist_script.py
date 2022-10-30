import pandas as pd

from activation_functions import ActivationFunctions
from error_functions import ErrorFunctions
from mnist import MNIST
import numpy as np

from network import Network
from network_enums import ProblemType

mndata = MNIST('./mnist')

images, labels = mndata.load_training()
# or
# images, labels = mndata.load_testing()

# Problems with adapting PyCharm Console width to the width of samples
# desired_width = 320
# pd.set_option('display.width', desired_width)
# pd.set_option('display.max_columns', 20)
# np.set_printoptions(linewidth=desired_width)

index = 1
print(mndata.display(images[index]))

# Numbers are stored in images table as 28*28 arrays. Each element of the array ranges from 0 to 255 indicating grey scale
# Labels store number presented in i-th image

# Questions:
# 1. How to pass input?
# 2. How to adapt current solution?
# 3. How to adjust network parameters?
# NOTE: Take a look at mnist page and table with accuracy described: http://yann.lecun.com/exdb/mnist/
# Idea: Start with 2-layer NN, 300 hidden units, mean square error and see what happens?
# What about activation function?

problem_type = ProblemType.CLASSIFICATION
use_bias = True
hidden_layers_count = 2
hidden_layer_neurons_count = 300
activation_function = ActivationFunctions.sigmoid
error_function = ErrorFunctions.mean_squared_error
initial_seed = 1
learning_rate = 0.01
epoch_max = 300
visualize_epochs = False
input_size = 28*28
output_layer_size = 10

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
