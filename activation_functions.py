from math import e
from cmath import tanh
from typing import Callable, Optional

ActivationFunctionType = Callable[[float, Optional[bool]], float]

class ActivationFunctions:
    @staticmethod
    def sigmoid(x_value: float, derivative: bool = False) -> float:
        if derivative:
            return x_value * (1.0 - x_value)
        return 1.0 / (1.0 + e ** -x_value)

    @staticmethod
    def tanh(x_value: float, derivative = False) -> float:
        if derivative:
            return 1 - tanh(x_value) ** 2
        return tanh(x_value)
