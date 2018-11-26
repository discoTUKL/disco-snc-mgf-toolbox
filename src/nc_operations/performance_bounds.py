"""Performance bounds"""

from math import exp, inf, log

from library.exceptions import ParameterOutOfBounds
from library.helper_functions import get_q
from nc_processes.arrival import Arrival
from nc_processes.service import Service


def backlog_prob(arr: Arrival,
                 ser: Service,
                 theta: float,
                 backlog_value: float,
                 tau=1.0,
                 indep=True,
                 p=1.0) -> float:
    """Implements stationary bound method"""
    if indep:
        p = 1.0

    q = get_q(p=p, indep=indep)

    rho_a_p = arr.rho(theta=p * theta)
    sigma_a_p = arr.sigma(theta=p * theta)
    rho_s_q = ser.rho(theta=q * theta)
    sigma_s_q = ser.sigma(theta=q * theta)

    if rho_a_p >= rho_s_q:
        raise ParameterOutOfBounds(
            f"The arrivals' rho {rho_a_p} has to be smaller than"
            f"the service's rho {rho_s_q}")

    rho_arr_ser = rho_a_p - rho_s_q
    sigma_arr_ser = sigma_a_p + sigma_s_q

    try:
        if arr.is_discrete():
            return exp(-theta * backlog_value) * exp(
                theta * sigma_arr_ser) / (1 - exp(theta * rho_arr_ser))

        else:
            return exp(-theta * backlog_value) * exp(
                theta * (rho_a_p * tau + sigma_arr_ser)) / (
                           1 - exp(theta * tau * rho_arr_ser))

    except ZeroDivisionError:
        return inf


def backlog(arr: Arrival,
            ser: Service,
            theta: float,
            prob_b: float,
            tau=1.0,
            indep=True,
            p=1.0) -> float:
    """Implements stationary bound method"""
    if indep:
        p = 1.0

    q = get_q(p=p, indep=indep)

    rho_a_p = arr.rho(theta=p * theta)
    sigma_a_p = arr.sigma(theta=p * theta)
    rho_s_q = ser.rho(theta=q * theta)
    sigma_s_q = ser.sigma(theta=q * theta)

    if rho_a_p >= rho_s_q:
        raise ParameterOutOfBounds(
            f"The arrivals' rho {rho_a_p} has to be smaller than"
            f"the service's rho {rho_s_q}")

    rho_arr_ser = rho_a_p - rho_s_q
    sigma_arr_ser = sigma_a_p + sigma_s_q

    if arr.is_discrete():
        log_part = log(prob_b * (1 - exp(theta * rho_arr_ser)))

        return sigma_arr_ser - log_part / theta

    else:
        log_part = log(prob_b * (1 - exp(theta * tau * rho_arr_ser)))

        return tau * rho_a_p + sigma_arr_ser - log_part / theta


def delay_prob(arr: Arrival,
               ser: Service,
               theta: float,
               delay_value: int,
               tau=1.0,
               indep=True,
               p=1.0) -> float:
    """Implements stationary bound method"""
    if indep:
        p = 1.0

    q = get_q(p=p, indep=indep)

    rho_a_p = arr.rho(theta=p * theta)
    sigma_a_p = arr.sigma(theta=p * theta)
    rho_s_q = ser.rho(theta=q * theta)
    sigma_s_q = ser.sigma(theta=q * theta)

    if rho_a_p >= rho_s_q:
        raise ParameterOutOfBounds(
            f"The arrivals' rho {rho_a_p} has to be smaller than"
            f"the service's rho {rho_s_q}")

    rho_arr_ser = rho_a_p - rho_s_q
    sigma_arr_ser = sigma_a_p + sigma_s_q

    try:
        if arr.is_discrete():
            return exp(-theta * rho_s_q * delay_value) * exp(
                theta * sigma_arr_ser) / (1 - exp(theta * rho_arr_ser))
        else:
            return exp(-theta * rho_s_q * delay_value) * exp(
                theta * (rho_a_p * tau + sigma_arr_ser)) / (
                           1 - exp(theta * tau * rho_arr_ser))

    except ZeroDivisionError:
        return inf


def delay(arr: Arrival,
          ser: Service,
          theta: float,
          prob_d: float,
          tau=1.0,
          indep=True,
          p=1.0) -> float:
    """Implements stationary bound method"""
    if indep:
        p = 1.0

    q = get_q(p=p, indep=indep)

    rho_a_p = arr.rho(theta=p * theta)
    sigma_a_p = arr.sigma(theta=p * theta)
    rho_s_q = ser.rho(theta=q * theta)
    sigma_s_q = ser.sigma(theta=q * theta)

    if rho_a_p >= rho_s_q:
        raise ParameterOutOfBounds(
            f"The arrivals' rho {rho_a_p} has to be smaller than"
            f"the service's rho {rho_s_q}")

    rho_arr_ser = rho_a_p - rho_s_q
    sigma_arr_ser = sigma_a_p + sigma_s_q

    if arr.is_discrete():
        log_part = log(prob_d * (1 - exp(theta * rho_arr_ser)))

        return (sigma_arr_ser - log_part / theta) / rho_s_q

    else:
        log_part = log(prob_d * (1 - exp(theta * tau * rho_arr_ser)))

        return (tau * rho_a_p + sigma_arr_ser - log_part / theta) * rho_s_q


def output(arr: Arrival,
           ser: Service,
           theta: float,
           delta_time: int,
           indep=True,
           p=1.0) -> float:
    """Implements stationary bound method"""
    if indep:
        p = 1.0

    q = get_q(p=p, indep=indep)

    rho_a_p = arr.rho(theta=p * theta)
    sigma_a_p = arr.sigma(theta=p * theta)
    rho_s_q = ser.rho(theta=q * theta)
    sigma_s_q = ser.sigma(theta=q * theta)

    if rho_a_p >= rho_s_q:
        raise ParameterOutOfBounds(
            f"The arrivals' rho {rho_a_p} has to be smaller than"
            f"the service's rho {rho_s_q}")

    rho_arr_ser = rho_a_p - rho_s_q
    sigma_arr_ser = sigma_a_p + sigma_s_q

    try:
        if arr.is_discrete():
            return exp(theta * rho_a_p * delta_time) * exp(
                theta * sigma_arr_ser) / (1 - exp(theta * rho_arr_ser))

        else:
            return exp(theta * rho_a_p * (delta_time + 1)) * exp(
                theta * sigma_arr_ser) / (1 - exp(theta * rho_arr_ser))

    except ZeroDivisionError:
        return inf
