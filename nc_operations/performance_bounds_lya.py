"""Implements new Lyapunov Output Bound"""

from math import exp, inf

from library.exceptions import ParameterOutOfBounds
from library.helper_functions import is_equal
from nc_processes.arrival import Arrival
from nc_processes.service import Service


def output_lya(arr: Arrival,
               ser: Service,
               theta: float,
               delta_time: int,
               l_lya=1.0) -> float:
    """Implements stationary bound method"""
    if l_lya < 1.0:
        l_lya = 1.0
        # raise ParameterOutOfBounds("l must be >= 1")

    l_theta = l_lya * theta

    if arr.rho(l_theta) >= -ser.rho(l_theta):
        raise ParameterOutOfBounds(
            "The arrivals' RHO_SINGLE {0} has to be smaller than"
            "the service's RHO_SINGLE {1}".format(
                arr.rho(theta), -ser.rho(theta)))

    sigma_l_arr_ser = arr.sigma(l_theta) + ser.sigma(l_theta)
    rho_l_arr_ser = arr.rho(l_theta) + ser.rho(l_theta)

    numerator = exp(theta * (arr.rho(l_theta) * delta_time + sigma_l_arr_ser))
    denominator = (1 - exp(l_theta * rho_l_arr_ser))**(1 / l_lya)

    try:
        return numerator / denominator

    except ZeroDivisionError:
        return inf


def output_lya_t(arr: Arrival,
                 ser: Service,
                 theta: float,
                 tt: int,
                 ss: int,
                 l_lya=1.0) -> float:
    """Implements time dependent method"""
    if l_lya < 1.0:
        l_lya = 1.0
        # raise ParameterOutOfBounds("l must be >= 1")

    l_theta = l_lya * theta

    sigma_l_arr_ser = arr.sigma(l_theta) + ser.sigma(l_theta)
    rho_l_arr_ser = arr.rho(l_theta) + ser.rho(l_theta)

    if is_equal(arr.rho(l_theta), -ser.rho(l_theta)):
        return exp(theta *
                   (arr.rho(l_theta) *
                    (tt - ss) + sigma_l_arr_ser)) * (ss + 1)**(1 / l_lya)

    elif arr.rho(l_theta) > -ser.rho(l_theta):
        numerator = exp(
            theta *
            (arr.rho(l_theta) * tt + ser.rho(l_theta) * ss + sigma_l_arr_ser))
        denominator = 1 - exp(-l_theta * rho_l_arr_ser)**(1 / l_lya)

        return numerator / denominator

    else:
        return output_lya(
            arr=arr, ser=ser, theta=theta, delta_time=tt - ss, l_lya=l_lya)


def delay_prob_lya(arr: Arrival,
                   ser: Service,
                   theta: float,
                   delay: int,
                   l_lya=1.0) -> float:
    """Implements stationary bound method"""
    if l_lya < 1.0:
        l_lya = 1.0
        # raise ParameterOutOfBounds("l must be >= 1")

    l_theta = l_lya * theta

    if arr.rho(l_theta) >= -ser.rho(l_theta):
        raise ParameterOutOfBounds(
            "The arrivals' RHO_SINGLE {0} has to be smaller than"
            "the service's RHO_SINGLE {1}".format(
                arr.rho(theta), -ser.rho(theta)))

    sigma_l_arr_ser = arr.sigma(l_theta) + ser.sigma(l_theta)
    rho_l_arr_ser = arr.rho(l_theta) + ser.rho(l_theta)

    numerator = exp(theta * (ser.rho(l_theta) * delay + sigma_l_arr_ser))
    denominator = (1 - exp(l_theta * rho_l_arr_ser))**(1 / l_lya)

    try:
        return numerator / denominator

    except ZeroDivisionError:
        return inf


def delay_prob_lya_t(arr: Arrival,
                     ser: Service,
                     theta: float,
                     delay: int,
                     tt: int,
                     l_lya=1.0) -> float:
    """Implements time dependent method"""
    if l_lya < 1.0:
        l_lya = 1.0
        # raise ParameterOutOfBounds("l must be >= 1")

    l_theta = l_lya * theta

    sigma_l_arr_ser = arr.sigma(l_theta) + ser.sigma(l_theta)
    rho_l_arr_ser = arr.rho(l_theta) + ser.rho(l_theta)

    if is_equal(arr.rho(l_theta), -ser.rho(l_theta)):
        return exp(theta * (ser.rho(l_theta) * delay + sigma_l_arr_ser)) * (
            tt + 1)**(1 / l_lya)

    elif arr.rho(l_theta) > -ser.rho(l_theta):
        numerator = exp(theta * (arr.rho(l_theta) * tt + ser.rho(l_theta) *
                                 (tt + delay) + sigma_l_arr_ser))
        denominator = 1 - exp(-l_theta * rho_l_arr_ser)**(1 / l_lya)

        return numerator / denominator

    else:
        return delay_prob_lya(
            arr=arr, ser=ser, theta=theta, delay=delay, l_lya=l_lya)


def output_lya_discretized(arr: Arrival,
                           ser: Service,
                           theta: float,
                           delta_time: int,
                           l_lya=1.0) -> float:
    """Implements stationary bound method"""
    if l_lya < 1.0:
        l_lya = 1.0
        # raise ParameterOutOfBounds("l must be >= 1")

    l_theta = l_lya * theta

    if arr.rho(l_theta) >= -ser.rho(l_theta):
        raise ParameterOutOfBounds(
            "The arrivals' RHO_SINGLE {0} has to be smaller than"
            "the service's RHO_SINGLE {1}".format(
                arr.rho(theta), -ser.rho(theta)))

    sigma_l_arr_ser = arr.sigma(l_theta) + ser.sigma(l_theta)
    rho_l_arr_ser = arr.rho(l_theta) + ser.rho(l_theta)

    numerator = exp(
        theta * (arr.rho(l_theta) * (delta_time + 1) + sigma_l_arr_ser))
    denominator = (1 - exp(l_theta * rho_l_arr_ser))**(1 / l_lya)

    try:
        return numerator / denominator

    except ZeroDivisionError:
        return inf
