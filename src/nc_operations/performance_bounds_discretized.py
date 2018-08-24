"""Performance bounds for continuous Process (need discretization)"""

from math import inf, log

from library.exceptions import ParameterOutOfBounds
from library.helper_functions import get_q, mgf
from nc_processes.arrival import Arrival
from nc_processes.service import Service


def backlog_prob_discretized(arr: Arrival,
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

    if rho_a_p >= -rho_s_q:
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(rho_a_p, -rho_s_q))

    return mgf(
        theta=theta, x=-backlog_value + rho_a_p * tau + sigma_a_p +
        sigma_s_q) / (1 - mgf(theta=theta, x=tau * (rho_a_p + rho_s_q)))


def backlog_discretized(arr: Arrival,
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

    if rho_a_p >= -rho_s_q:
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(rho_a_p, -rho_s_q))

    log_part = log(
        prob_b * (1 - mgf(theta=theta, x=tau * (rho_a_p + rho_s_q))))

    return tau * rho_a_p + sigma_a_p + sigma_s_q - log_part / theta


def delay_prob_discretized(arr: Arrival,
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

    if rho_a_p >= -rho_s_q:
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(rho_a_p, -rho_s_q))

    try:
        return mgf(
            theta=theta,
            x=rho_a_p * tau + sigma_a_p + sigma_s_q + rho_s_q * delay_value
        ) / (1 - mgf(theta=theta, x=tau * (rho_a_p + rho_s_q)))

    except ZeroDivisionError:
        return inf


def delay_discretized(arr: Arrival,
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

    if rho_a_p >= -rho_s_q:
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(rho_a_p, -rho_s_q))

    log_part = log(
        prob_d * (1 - mgf(theta=theta, x=tau * (rho_a_p + rho_s_q))))

    return (log_part / theta -
            (tau * rho_a_p + sigma_a_p + sigma_s_q)) / rho_s_q


def output_discretized(arr: Arrival,
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

    if rho_a_p >= -rho_s_q:
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(rho_a_p, -rho_s_q))

    try:
        return mgf(
            theta=theta, x=rho_a_p * (delta_time + 1) + sigma_a_p +
            sigma_s_q) / (1 - mgf(theta=theta, x=rho_a_p + rho_s_q))

    except ZeroDivisionError:
        return inf
