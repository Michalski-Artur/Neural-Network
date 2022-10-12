# Imports
import csv
from cmath import tanh
from math import e
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Activation functions

def sigmoid_func(x):
    return 1 / (1 + e ** -x)

def tanh_func(x):
    return tanh(x)

def sign_func(x):
    return -1 if (x < 0) else 1

# Network params

seed = 1
layers_no = 3
neurons_nos = [2, 3, 1]
is_bias_present = True

# print(sigmoid_func(1))
# print(sign_func(-3))
# print(sign_func(3))

# Reading data from file

def read_data(filename):
    return pd.read_csv(filename)

# Visualize training set

def visualize_data(df):
    first_class = df[(df['cls']==1)]
    ax = first_class.plot.scatter(x='x',y='y',c='g', label='First class')
    second_class = df[(df['cls']==2)]
    second_class.plot.scatter(x='x',y='y',ax=ax, c='r', label='Second class')
    plt.title('Input data')
    plt.show()

#df = read_data("classification/data.simple.test.100.csv")
df = read_data("classification/data.three_gauss.train.10000.csv")
visualize_data(df)

# Visualize classification effects - use colors for network classification and shapes for actual classes

# TODO:
# Initialize network using params and provided activation function - with random weight generation
# Implement weights updating algorithm using backwards error propagation algorithm
# Visualize weights and propagated error for each epoch



# Think over specifying classes for neuron, layer, network?
w = np.random.random_sample(size = 4)
print(w)