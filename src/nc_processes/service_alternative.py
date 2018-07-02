"""Class for all service processes that cannot be described via (sigma, rho)"""

from math import inf
from library.helper_functions import mgf


def constant_rate_alternative(theta: float, delta_time: int,
                              rate: float) -> float:
    if theta <= 0:
        return inf
        # raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

    return mgf(theta=theta, x=-rate * delta_time)
