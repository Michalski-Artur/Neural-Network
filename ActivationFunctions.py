from math import e
from cmath import tanh

class ActivationFunctions:
    @staticmethod
    def sigmoid(x_value: float, derivative = False) -> float:
        if derivative:
            return ActivationFunctions.sigmoid(x_value) * (1 - ActivationFunctions.sigmoid(x_value))
        return 1 / (1 + e ** -x_value)

    @staticmethod
    def tanh(x_value: float, derivative = False) -> float:
        if derivative:
            return 1 - tanh(x_value) ** 2
        return tanh(x_value)
