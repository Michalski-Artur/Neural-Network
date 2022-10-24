from typing import Callable, Optional
from math import e
import numpy as np

ActivationFunctionType = Callable[[float, Optional[bool]], float]

class ActivationFunctions:

    @staticmethod
    def identity(x_value: float, derivative: bool = False) -> float:
        return 1.0 if derivative else x_value

    @staticmethod
    def sigmoid(x_value: float, derivative: bool = False) -> float:
        # x_value is sigmoid(x) for derivative
        if derivative:
            return x_value * (1.0 - x_value)
        return 1.0 / (1.0 + e ** -x_value)

    @staticmethod
    def tanh(x_value: float, derivative = False) -> float:
        # x_value is tanh(x) for derivative
        if derivative:
            return 1 - x_value ** 2
        return np.tanh(x_value).real

    @staticmethod
    def softmax_vector(x_vector: list[float]) -> list[float]:
        val = np.exp(x_vector)
        return list(val / val.sum())
