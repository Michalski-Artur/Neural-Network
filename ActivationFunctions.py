from math import e
from cmath import tanh


class ActivationFunctions:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + e ** -x)

    @staticmethod
    def sigmoid_derivative(x):
        return 1 / (1 + e ** -x)

    @staticmethod
    def tanh(x):
        return tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - tanh(x) ** 2
