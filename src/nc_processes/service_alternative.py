"""Class for all service processes that cannot be described via (sigma, rho)"""
from math import exp

from library.exceptions import ParameterOutOfBounds


def mgf_const_rate(theta: float, delta_time: int, rate: float) -> float:
    if theta <= 0:
        raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

    return exp(-theta * rate * delta_time)


def expect_const_rate(delta_time: int, rate: float) -> float:
    return delta_time * rate
