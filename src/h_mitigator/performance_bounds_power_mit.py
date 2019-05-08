"""Implements new Lyapunov Output Bound"""

from math import exp, inf
from warnings import warn

from nc_arrivals.arrival import Arrival
from nc_operations.get_sigma_rho import get_sigma_rho
from nc_operations.stability_check import stability_check
from nc_server.server import Server


def output_power_mit(arr: Arrival,
                     ser: Server,
                     theta: float,
                     delta_time: int,
                     l_power=1.0) -> float:
    """Implements stationary bound method"""
    if l_power < 1.0:
        l_power = 1.0
        # raise ParameterOutOfBounds("l must be >= 1")

    l_theta = l_power * theta

    stability_check(arr=arr, ser=ser, theta=l_theta, indep=True, p=1.0)
    sigma_l_sum, rho_l_diff = get_sigma_rho(arr=arr,
                                            ser=ser,
                                            theta=l_theta,
                                            indep=True,
                                            p=1.0)

    if arr.is_discrete():
        numerator = exp(theta * arr.rho(theta=l_theta) * delta_time) * exp(
            theta * sigma_l_sum)
        denominator = (1 - exp(l_theta * rho_l_diff))**(1 / l_power)

    else:
        numerator = exp(theta * arr.rho(theta=l_theta) *
                        (delta_time + 1)) * exp(theta * sigma_l_sum)
        denominator = (1 - exp(l_theta * rho_l_diff))**(1 / l_power)

    try:
        return numerator / denominator

    except ZeroDivisionError:
        return inf


def delay_prob_power_mit(arr: Arrival,
                         ser: Server,
                         theta: float,
                         delay: int,
                         l_power=1.0) -> float:
    """Implements stationary bound method"""
    if l_power < 1.0:
        l_power = 1.0
        # raise ParameterOutOfBounds("l must be >= 1")

    l_theta = l_power * theta

    stability_check(arr=arr, ser=ser, theta=l_theta, indep=True, p=1.0)
    sigma_l_sum, rho_l_diff = get_sigma_rho(arr=arr,
                                            ser=ser,
                                            theta=l_theta,
                                            indep=True,
                                            p=1.0)

    if not arr.is_discrete():
        warn("discretized version is not implemented")

    numerator = exp(theta * ser.rho(theta=l_theta) * delay) * exp(
        theta * sigma_l_sum)
    denominator = (1 - exp(l_theta * rho_l_diff))**(1 / l_power)

    try:
        return numerator / denominator

    except ZeroDivisionError:
        return inf
