"""Class for all service processes that cannot be described via (sigma, rho)"""

from library.exceptions import ParameterOutOfBounds
from library.helper_functions import mgf


def mgf_const_rate(theta: float, delta_time: int, rate: float) -> float:
    if theta <= 0:
        raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

    return mgf(theta=theta, x=-rate * delta_time)


def expect_const_rate(theta: float, delta_time: int, rate: float) -> float:
    if theta <= 0:
        raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

    return theta * delta_time * rate


def long_term_const_rate(rate: float) -> float:
    return rate
