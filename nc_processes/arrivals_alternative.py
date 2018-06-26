"""Class for all arrival processes that cannot be described via (sigma, rho)"""

from math import exp, inf

from library.exceptions import ParameterOutOfBounds


def fbm(theta: float, delta_time: int, lamb: float, sigma: float,
        hurst: float) -> float:
    if theta <= 0:
        raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

    if sigma <= 0:
        raise ParameterOutOfBounds("sigma = {0} must be > 0".format(sigma))

    if hurst >= 1 or hurst <= 0:
        raise ParameterOutOfBounds(
            "Hurst = {0} must be in (0, 1)".format(hurst))

    try:
        return exp(lamb * theta * delta_time +
                   (0.5 * (sigma * theta)**2) * delta_time**(2 * hurst))
    except OverflowError:
        return inf


def regulated_alternative(theta: float,
                          delta_time: int,
                          sigma_single: float,
                          rho_single: float,
                          n=1) -> float:
    if theta <= 0:
        return inf
        # raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

    try:
        return 1 + rho_single * delta_time / (
            sigma_single + rho_single * delta_time) * (
                exp(n * theta * (sigma_single + rho_single * delta_time)) - 1)
    except OverflowError:
        return inf
