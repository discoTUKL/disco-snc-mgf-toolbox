"""Implements new Lyapunov Output Bound"""

from math import exp, inf

from library.exceptions import ParameterOutOfBounds
from nc_processes.arrival import Arrival
from nc_processes.service import Service


def output_power(arr: Arrival,
                 ser: Service,
                 theta: float,
                 delta_time: int,
                 l_power=1.0) -> float:
    """Implements stationary bound method"""
    if l_power < 1.0:
        l_power = 1.0
        # raise ParameterOutOfBounds("l must be >= 1")

    l_theta = l_power * theta

    if arr.rho(theta=l_theta) >= ser.rho(theta=l_theta):
        raise ParameterOutOfBounds(
            f"The arrivals' rho {arr.rho(theta)} has to be smaller than"
            f"the service's rho {ser.rho(theta)}")

    sigma_l_arr_ser = arr.sigma(theta=l_theta) + ser.sigma(theta=l_theta)
    rho_l_arr_ser = arr.rho(theta=l_theta) - ser.rho(theta=l_theta)

    if arr.is_discrete():
        numerator = exp(theta * arr.rho(theta=l_theta) * delta_time) * exp(
            theta * sigma_l_arr_ser)
        denominator = (1 - exp(l_theta * rho_l_arr_ser))**(1 / l_power)

    else:
        numerator = exp(
            theta * arr.rho(theta=l_theta) * (delta_time + 1)) * exp(
            theta * sigma_l_arr_ser)
        denominator = (1 - exp(l_theta * rho_l_arr_ser)) ** (1 / l_power)

    try:
        return numerator / denominator

    except ZeroDivisionError:
        return inf


def delay_prob_power(arr: Arrival,
                     ser: Service,
                     theta: float,
                     delay: int,
                     l_power=1.0) -> float:
    """Implements stationary bound method"""
    if l_power < 1.0:
        l_power = 1.0
        # raise ParameterOutOfBounds("l must be >= 1")

    l_theta = l_power * theta

    if arr.rho(theta=l_theta) >= ser.rho(theta=l_theta):
        raise ParameterOutOfBounds(
            f"The arrivals' rho {arr.rho(theta)} has to be smaller than"
            f"the service's rho {ser.rho(theta)}")

    sigma_l_arr_ser = arr.sigma(theta=l_theta) + ser.sigma(theta=l_theta)
    rho_l_arr_ser = arr.rho(theta=l_theta) - ser.rho(theta=l_theta)

    numerator = exp(theta * ser.rho(theta=l_theta) * delay) * exp(
        theta * sigma_l_arr_ser)
    denominator = (1 - exp(l_theta * rho_l_arr_ser))**(1 / l_power)

    try:
        return numerator / denominator

    except ZeroDivisionError:
        return inf
