"""Performance bounds"""

from math import exp, inf, log

from library.exceptions import ParameterOutOfBounds
from library.helper_functions import is_equal, mgf
from nc_processes.arrival import Arrival
from nc_processes.service import Service


def backlog_prob(arr: Arrival, ser: Service, theta: float,
                 backlog_value: float) -> float:
    """Implements stationary bound method"""

    if arr.rho(theta=theta) >= -ser.rho(theta=theta):
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(
                arr.rho(theta=theta), -ser.rho(theta=theta)))

    sigma_arr_ser = arr.sigma(theta=theta) + ser.sigma(theta=theta)
    rho_arr_ser = arr.rho(theta=theta) + ser.rho(theta=theta)

    return mgf(
        theta=theta, x=-backlog_value + sigma_arr_ser) / (
            1 - mgf(theta=theta, x=rho_arr_ser))


def backlog_prob_t(arr: Arrival, ser: Service, theta: float, tt: int,
                   backlog_value: float) -> float:
    """Implements time dependent method"""

    sigma_arr_ser = arr.sigma(theta=theta) + ser.sigma(theta=theta)
    rho_arr_ser = arr.rho(theta=theta) + ser.rho(theta=theta)

    if is_equal(arr.rho(theta=theta), -ser.rho(theta=theta)):
        return exp(theta * (-backlog_value + sigma_arr_ser)) * (tt + 1)

    elif arr.rho(theta=theta) > -ser.rho(theta=theta):
        return exp(theta *
                   (-backlog_value + rho_arr_ser * tt + sigma_arr_ser)) / (
                       1 - exp(-theta * rho_arr_ser))

    else:
        return backlog_prob(
            arr=arr, ser=ser, theta=theta, backlog_value=backlog_value)


def backlog(arr: Arrival, ser: Service, theta: float, prob_b: float) -> float:
    """Implements stationary bound method"""

    if arr.rho(theta=theta) >= -ser.rho(theta=theta):
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(
                arr.rho(theta=theta), -ser.rho(theta=theta)))

    sigma_arr_ser = arr.sigma(theta=theta) + ser.sigma(theta=theta)
    rho_arr_ser = arr.rho(theta=theta) + ser.rho(theta=theta)

    log_part = log(prob_b * (1 - mgf(theta=theta, x=rho_arr_ser)))

    return sigma_arr_ser - log_part / theta


def backlog_t(arr: Arrival, ser: Service, theta: float, prob_b: float,
              tt: int) -> float:
    """Implements time dependent method"""

    sigma_arr_ser = arr.sigma(theta=theta) + ser.sigma(theta=theta)
    rho_arr_ser = arr.rho(theta=theta) + ser.rho(theta=theta)

    if is_equal(arr.rho(theta=theta), -ser.rho(theta=theta)):
        log_part = log(prob_b / (tt + 1))

        return sigma_arr_ser - log_part / theta

    elif arr.rho(theta=theta) > -ser.rho(theta=theta):
        log_part = log(prob_b * (1 - exp(-theta * rho_arr_ser)))

        return rho_arr_ser * tt + sigma_arr_ser - log_part / theta

    else:
        return backlog(arr=arr, ser=ser, theta=theta, prob_b=prob_b)


def delay_prob(arr: Arrival, ser: Service, theta: float,
               delay_value: int) -> float:
    """Implements stationary bound method"""

    if arr.rho(theta=theta) >= -ser.rho(theta=theta):
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(
                arr.rho(theta=theta), -ser.rho(theta=theta)))

    sigma_arr_ser = arr.sigma(theta=theta) + ser.sigma(theta=theta)
    rho_arr_ser = arr.rho(theta=theta) + ser.rho(theta=theta)

    return mgf(
        theta=theta, x=sigma_arr_ser + ser.rho(theta=theta) * delay_value) / (
            1 - mgf(theta=theta, x=rho_arr_ser))


def delay_prob_t(arr: Arrival, ser: Service, theta: float, tt: int,
                 delay_value: int) -> float:
    """Implements time dependent method"""

    sigma_arr_ser = arr.sigma(theta=theta) + ser.sigma(theta=theta)
    rho_arr_ser = arr.rho(theta=theta) + ser.rho(theta=theta)

    if is_equal(arr.rho(theta=theta), -ser.rho(theta=theta)):
        return exp(
            theta *
            (ser.rho(theta=theta) * delay_value + sigma_arr_ser)) * (tt + 1)

    elif arr.rho(theta=theta) > -ser.rho(theta=theta):
        return exp(theta * (arr.rho(theta=theta) * tt + ser.rho(theta=theta) *
                            (tt + delay_value) + sigma_arr_ser)) / (
                                1 - exp(-theta * rho_arr_ser))

    else:
        return delay_prob(
            arr=arr, ser=ser, theta=theta, delay_value=delay_value)


def delay(arr: Arrival, ser: Service, theta: float, prob_d: float) -> float:
    """Implements stationary bound method"""

    if arr.rho(theta=theta) >= -ser.rho(theta=theta):
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(
                arr.rho(theta=theta), -ser.rho(theta=theta)))

    sigma_arr_ser = arr.sigma(theta=theta) + ser.sigma(theta=theta)
    rho_arr_ser = arr.rho(theta=theta) + ser.rho(theta=theta)

    log_part = log(prob_d * (1 - mgf(theta=theta, x=rho_arr_ser)))

    return (log_part / theta - sigma_arr_ser) / ser.rho(theta=theta)


def delay_t(arr: Arrival, ser: Service, theta: float, prob_d: float,
            tt: int) -> float:
    """Implements time dependent method"""

    sigma_arr_ser = arr.sigma(theta=theta) + ser.sigma(theta=theta)
    rho_arr_ser = arr.rho(theta=theta) + ser.rho(theta=theta)

    if arr.rho(theta=theta) == -ser.rho(theta=theta):
        log_part = log(prob_d / (tt + 1))

        return (log_part / theta - sigma_arr_ser) / ser.rho(theta=theta)

    elif arr.rho(theta=theta) > -ser.rho(theta=theta):
        log_part = log(prob_d * (1 - exp(-theta * rho_arr_ser)))

        return log_part / theta - (
            rho_arr_ser * tt + sigma_arr_ser) / ser.rho(theta=theta)

    else:
        return delay(arr=arr, ser=ser, theta=theta, prob_d=prob_d)


def output(arr: Arrival, ser: Service, theta: float, delta_time: int) -> float:
    """Implements stationary bound method"""

    if arr.rho(theta=theta) >= -ser.rho(theta=theta):
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(
                arr.rho(theta=theta), -ser.rho(theta=theta)))

    sigma_arr_ser = arr.sigma(theta=theta) + ser.sigma(theta=theta)
    rho_arr_ser = arr.rho(theta=theta) + ser.rho(theta=theta)

    numerator = mgf(
        theta=theta, x=arr.rho(theta=theta) * delta_time + sigma_arr_ser)
    denominator = 1 - mgf(theta=theta, x=rho_arr_ser)

    try:
        return numerator / denominator

    except ZeroDivisionError:
        return inf


def output_t(arr: Arrival, ser: Service, theta: float, tt: int,
             ss: int) -> float:
    """Implements time dependent method"""

    sigma_arr_ser = arr.sigma(theta=theta) + ser.sigma(theta=theta)
    rho_arr_ser = arr.rho(theta=theta) + ser.rho(theta=theta)

    if is_equal(arr.rho(theta=theta), -ser.rho(theta=theta)):
        return exp(theta * (arr.rho(theta=theta) *
                            (tt - ss) + sigma_arr_ser)) * (ss + 1)

    elif arr.rho(theta=theta) > -ser.rho(theta=theta):
        return exp(theta * (arr.rho(theta=theta) * tt +
                            ser.rho(theta=theta) * ss + sigma_arr_ser) / 1 -
                   exp(-theta * rho_arr_ser))

    else:
        return output(arr=arr, ser=ser, theta=theta, delta_time=tt - ss)
