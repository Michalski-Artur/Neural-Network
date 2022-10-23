from cmath import log
from typing import Callable, Optional

ErrorFunctionType = Callable[[float, float, Optional[bool]], float]

class ErrorFunctions:
    @staticmethod
    def cross_entropy_error(calculated: float, expected: float, derivative: bool = False) -> float:
        if derivative:
            if calculated == 1.0:
                return 0.0 if expected == 1.0 else 1.0
            return -(expected / calculated - (1 - expected) / (1 - calculated)).real
        else:
            return (expected * log(calculated) + (1 - expected) * log(1 - calculated)).real

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
