"""Performance bounds"""

import math

from nc_arrivals.arrival import Arrival
from nc_operations.get_sigma_rho import get_sigma_rho
from nc_operations.stability_check import stability_check
from nc_server.server import Server
from utils.exceptions import IllegalArgumentError
from utils.helper_functions import get_q


def backlog_prob(arr: Arrival, ser: Server, theta: float, backlog_value: float, indep=True, p=1.0) -> float:
    """Implements stationary standard_bound method"""
    if indep:
        p = 1.0
        q = 1.0
    else:
        q = get_q(p=p)

    stability_check(arr=arr, ser=ser, theta=theta, indep=indep, p=p, q=q)
    sigma_sum, rho_diff = get_sigma_rho(arr=arr, ser=ser, theta=theta, indep=indep, p=p, q=q)

    try:
        if arr.is_discrete():
            return math.exp(-theta * backlog_value) * math.exp(theta * sigma_sum) / (-rho_diff * theta)
        else:
            tau_opt = 1 / (theta * ser.rho(theta=q * theta))
            return math.exp(-theta * backlog_value) * math.exp(
                theta * (ser.rho(theta=q * theta) * tau_opt + sigma_sum)) / (-rho_diff * theta * tau_opt)

    except ZeroDivisionError:
        return math.inf


def backlog(arr: Arrival, ser: Server, theta: float, prob_b: float, indep=True, p=1.0) -> float:
    """Implements stationary standard_bound method"""
    if prob_b < 0.0 or prob_b > 1.0:
        raise IllegalArgumentError(f"prob_b={prob_b} must be in (0,1)")

    if indep:
        p = 1.0
        q = 1.0
    else:
        q = get_q(p=p)

    stability_check(arr=arr, ser=ser, theta=theta, indep=indep, p=p, q=q)
    sigma_sum, rho_diff = get_sigma_rho(arr=arr, ser=ser, theta=theta, indep=indep, p=p, q=q)

    if arr.is_discrete():
        return sigma_sum - (log(prob_b * theta * (-rho_diff))) / theta
    else:
        tau_opt = 1 / (theta * ser.rho(theta=q * theta))
        log_part = math.log(prob_b * theta * tau_opt * (-rho_diff))
        return tau_opt * ser.rho(theta=q * theta) + sigma_sum - log_part / theta


def delay_prob(arr: Arrival, ser: Server, theta: float, delay_value: int, indep=True, p=1.0) -> float:
    """Implements stationary standard_bound method"""
    if indep:
        p = 1.0
        q = 1.0
    else:
        q = get_q(p=p)

    stability_check(arr=arr, ser=ser, theta=theta, indep=indep, p=p, q=q)
    sigma_sum, rho_diff = get_sigma_rho(arr=arr, ser=ser, theta=theta, indep=indep, p=p, q=q)

    try:
        if arr.is_discrete():
            return math.exp(-theta * ser.rho(theta=q * theta) * delay_value) * math.exp(
                theta * sigma_sum) / (-rho_diff * theta)
        else:
            tau_opt = 1 / (theta * ser.rho(theta=q * theta))
            return math.exp(-theta * ser.rho(theta=q * theta) * delay_value) * math.exp(
                theta * (ser.rho(theta=q * theta) * tau_opt + sigma_sum)) / (-rho_diff * theta * tau_opt)

    except ZeroDivisionError:
        return math.inf


def delay(arr: Arrival, ser: Server, theta: float, prob_d: float, indep=True, p=1.0) -> float:
    """Implements stationary standard_bound method"""
    if prob_d < 0.0 or prob_d > 1.0:
        raise IllegalArgumentError(f"prob_b={prob_d} must be in (0,1)")

    if indep:
        p = 1.0
        q = 1.0
    else:
        q = get_q(p=p)

    stability_check(arr=arr, ser=ser, theta=theta, indep=indep, p=p, q=q)
    sigma_sum, rho_diff = get_sigma_rho(arr=arr, ser=ser, theta=theta, indep=indep, p=p, q=q)

    if arr.is_discrete():
        log_part = math.log(prob_d * theta * (-rho_diff))
        return (sigma_sum - log_part / theta) / ser.rho(theta=q * theta)
    else:
        tau_opt = 1 / (theta * ser.rho(theta=q * theta))
        log_part = math.log(prob_d * theta * tau_opt * (-rho_diff))
        return (tau_opt * ser.rho(theta=q * theta) + sigma_sum - log_part / theta) / ser.rho(theta=q * theta)


def output(arr: Arrival, ser: Server, theta: float, delta_time: int, indep=True, p=1.0) -> float:
    """Implements stationary standard_bound method"""
    if indep:
        p = 1.0
        q = 1.0
    else:
        q = get_q(p=p)

    stability_check(arr=arr, ser=ser, theta=theta, indep=indep, p=p, q=q)
    sigma_sum, rho_diff = get_sigma_rho(arr=arr, ser=ser, theta=theta, indep=indep, p=p, q=q)

    try:
        if arr.is_discrete():
            return math.exp(theta * arr.rho(theta=p * theta) * delta_time) * math.exp(
                theta * sigma_sum) / (1 - math.exp(theta * rho_diff))

        else:
            return math.exp(theta * arr.rho(theta=p * theta) *
                            (delta_time + 1)) * math.exp(theta * sigma_sum) / (1 - math.exp(theta * rho_diff))

    except ZeroDivisionError:
        return math.inf
