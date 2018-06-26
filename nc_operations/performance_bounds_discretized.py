"""Performance bounds for continuous Process (need discretization)"""

from math import exp, inf, log

from library.exceptions import ParameterOutOfBounds
from nc_processes.arrival import Arrival
from nc_processes.service import Service


def backlog_prob_discretized(arr: Arrival,
                             ser: Service,
                             theta: float,
                             backlog: float,
                             tau=1.0) -> float:
    """Implements stationary bound method"""

    if arr.rho(theta) >= -ser.rho(theta):
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(arr.rho(theta), -ser.rho(theta)))

    sigma_arr_ser = arr.sigma(theta) + ser.sigma(theta)
    rho_arr_ser = arr.rho(theta) + ser.rho(theta)

    return exp(theta * (-backlog + arr.rho(theta) * tau + sigma_arr_ser)) / (
        1 - exp(theta * tau * rho_arr_ser))


def backlog_discretized(arr: Arrival,
                        ser: Service,
                        theta: float,
                        prob_b: float,
                        tau=1.0) -> float:
    """Implements stationary bound method"""

    if arr.rho(theta) >= -ser.rho(theta):
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(arr.rho(theta), -ser.rho(theta)))

    sigma_arr_ser = arr.sigma(theta) + ser.sigma(theta)
    rho_arr_ser = arr.rho(theta) + ser.rho(theta)

    log_part = log(prob_b * (1 - exp(theta * tau * rho_arr_ser)))

    return tau * arr.rho(theta) + sigma_arr_ser - log_part / theta


def delay_prob_discretized(arr: Arrival,
                           ser: Service,
                           theta: float,
                           delay: int,
                           tau=1.0) -> float:
    """Implements stationary bound method"""

    if arr.rho(theta) >= -ser.rho(theta):
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(arr.rho(theta), -ser.rho(theta)))

    sigma_arr_ser = arr.sigma(theta) + ser.sigma(theta)
    rho_arr_ser = arr.rho(theta) + ser.rho(theta)

    return exp(
        theta *
        (arr.rho(theta) * tau + sigma_arr_ser + ser.rho(theta) * delay)) / (
            1 - exp(theta * tau * rho_arr_ser))


def delay_discretized(arr: Arrival,
                      ser: Service,
                      theta: float,
                      prob_d: float,
                      tau=1.0) -> float:
    """Implements stationary bound method"""

    if arr.rho(theta) >= -ser.rho(theta):
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(arr.rho(theta), -ser.rho(theta)))

    sigma_arr_ser = arr.sigma(theta) + ser.sigma(theta)
    rho_arr_ser = arr.rho(theta) + ser.rho(theta)

    log_part = log(prob_d * (1 - exp(theta * tau * rho_arr_ser)))

    return (log_part / theta -
            (tau * arr.rho(theta) + sigma_arr_ser)) / ser.rho(theta)


def output_discretized(arr: Arrival, ser: Service, theta: float,
                       delta_time: int) -> float:
    """Implements stationary bound method"""

    if arr.rho(theta) >= -ser.rho(theta):
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(arr.rho(theta), -ser.rho(theta)))

    sigma_arr_ser = arr.sigma(theta) + ser.sigma(theta)
    rho_arr_ser = arr.rho(theta) + ser.rho(theta)

    numerator = exp(
        theta * (arr.rho(theta) * (delta_time + 1) + sigma_arr_ser))
    denominator = 1 - exp(theta * rho_arr_ser)

    try:
        return numerator / denominator

    except ZeroDivisionError:
        return inf
