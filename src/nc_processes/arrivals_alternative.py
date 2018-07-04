"""Class for all arrival processes that cannot be described via (sigma, rho)"""

from math import exp

from library.exceptions import ParameterOutOfBounds
from library.helper_functions import mgf


def fbm(theta: float, delta_time: int, lamb: float, sigma: float,
        hurst: float) -> float:
    if theta <= 0:
        raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

    if sigma <= 0:
        raise ParameterOutOfBounds("sigma = {0} must be > 0".format(sigma))

    if hurst <= 0 or hurst >= 1:
        raise ParameterOutOfBounds(
            "Hurst = {0} must be in (0, 1)".format(hurst))

    return exp(lamb * theta * delta_time +
               (0.5 * (sigma * theta) ** 2) * delta_time ** (2 * hurst))


def regulated_alternative(theta: float,
                          delta_time: int,
                          sigma_single: float,
                          rho_single: float,
                          n=1) -> float:
    if theta <= 0:
        raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

    rho_delta = rho_single * delta_time

    return 1 + rho_delta / (sigma_single + rho_delta) * (
            mgf(theta=theta, x=n * (sigma_single + rho_delta)) - 1)
