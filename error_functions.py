from typing import Callable, Optional
import numpy as np

ErrorFunctionType = Callable[[float, float, Optional[bool]], float]

class ErrorFunctions:
    @staticmethod
    def cross_entropy_error(calculated: float, expected: float, derivative: bool = False) -> float:
        # Only use for classification problems
        if derivative:
            if calculated == 1.0 or calculated == 0.0:
                return 0.0 if calculated == expected else 1.0
            return -(expected / calculated - (1 - expected) / (1 - calculated)).real
        else:
            if calculated == 1.0 or calculated == 0.0:
                return 0.0 if calculated == expected else 1.0
            return (expected * np.log(calculated) + (1 - expected) * np.log(1 - calculated)).real

    @staticmethod
    def mean_squared_error(calculated: float, expected: float, derivative: bool = False) -> float:
        if derivative:
            return calculated - expected
        else:
            return (calculated - expected) ** 2 / 2

    @staticmethod
    def mean_absolute_error(calculated: float, expected: float, derivative: bool = False) -> float:
        if derivative:
            return 1 if expected < calculated else -1 if expected > calculated else 0
        else:
            return abs(expected - calculated)
