from cmath import log
from typing import Callable, Optional

ErrorFunctionType = Callable[[float, float, Optional[bool]], float]

class ErrorFunctions:
    @staticmethod
    def cross_entropy(calculated: float, expected: float, derivative: bool = False) -> float:
        if derivative:
            return -expected / calculated + (1 - expected) / (1 - calculated)
        else:
            return expected * log(calculated) + (1 - expected) * log(1 - calculated)

    @staticmethod
    def mean_squared(calculated: float, expected: float, derivative: bool = False) -> float:
        if derivative:
            return calculated - expected
        else:
            return (expected - calculated) ** 2 / 2

    @staticmethod
    def mean_absolute(calculated: float, expected: float, derivative: bool = False) -> float:
        if derivative:
            return 1 if expected < calculated else -1 if expected > calculated else 0
        else:
            return abs(expected - calculated)
