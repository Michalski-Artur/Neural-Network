from math import e
from cmath import tanh


class ActivationFunctions:

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + e ** -x)

    @staticmethod
    def sigmoid_derivative(output):
        return output * (1.0 - output)

    @staticmethod
    def tanh(x):
        return tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - tanh(x) ** 2
