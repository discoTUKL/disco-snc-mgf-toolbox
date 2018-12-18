"""Class for all arrival processes that cannot be described via (sigma, rho)"""

from math import exp

from utils.exceptions import ParameterOutOfBounds


def mgf_fbm(theta: float, delta_time: int, lamb: float, sigma: float,
            hurst: float) -> float:
    if theta <= 0:
        raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

    if sigma <= 0:
        raise ParameterOutOfBounds(f"sigma = {sigma} must be > 0")

    if hurst <= 0 or hurst >= 1:
        raise ParameterOutOfBounds(f"Hurst = {hurst} must be in (0, 1)")

    return exp(lamb * theta * delta_time +
               (0.5 * (sigma * theta)**2) * delta_time**(2 * hurst))


def mgf_regulated_arrive(theta: float,
                         delta_time: int,
                         sigma_single: float,
                         rho_single: float,
                         n=1) -> float:
    if theta <= 0:
        raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

    rho_delta = rho_single * delta_time

    return 1 + rho_delta / (sigma_single + rho_delta) * (
        exp(theta * n * (sigma_single + rho_delta)) - 1)


def expect_dm1(delta_time: int, lamb: float) -> float:
    return delta_time / lamb


def var_dm1(delta_time: int, lamb: float) -> float:
    return delta_time / (lamb**2)
