"""Performance bounds"""

from math import exp, inf, log

from nc_arrivals.arrival import Arrival
from nc_operations.get_sigma_rho import get_sigma_rho
from nc_operations.stability_check import stability_check
from nc_server.server import Server
from utils.helper_functions import get_q


def backlog_prob(arr: Arrival,
                 ser: Server,
                 theta: float,
                 backlog_value: float,
                 tau=1.0,
                 indep=True,
                 p=1.0,
                 use_standard=True) -> float:
    """Implements stationary bound method"""
    if indep:
        p = 1.0

    stability_check(arr=arr, ser=ser, theta=theta, indep=indep, p=p)
    sigma_sum, rho_diff = get_sigma_rho(arr=arr,
                                        ser=ser,
                                        theta=theta,
                                        indep=indep,
                                        p=p)

    try:
        if not use_standard:
            if arr.is_discrete():
                return exp(-theta * backlog_value) * exp(
                    theta * sigma_sum) / (-theta * rho_diff)
            else:
                raise NotImplementedError

        if arr.is_discrete():
            return exp(-theta * backlog_value) * exp(
                theta * sigma_sum) / (1 - exp(theta * rho_diff))
        else:
            return exp(-theta * backlog_value) * exp(
                theta * (arr.rho(theta=p * theta) * tau + sigma_sum)) / (
                    1 - exp(theta * tau * rho_diff))

    except ZeroDivisionError:
        return inf


def backlog(arr: Arrival,
            ser: Server,
            theta: float,
            prob_b: float,
            tau=1.0,
            indep=True,
            p=1.0,
            use_standard=True) -> float:
    """Implements stationary bound method"""
    if indep:
        p = 1.0

    stability_check(arr=arr, ser=ser, theta=theta, indep=indep, p=p)
    sigma_sum, rho_diff = get_sigma_rho(arr=arr,
                                        ser=ser,
                                        theta=theta,
                                        indep=indep,
                                        p=p)

    if not use_standard:
        if arr.is_discrete():
            return sigma_sum - (log(-prob_b * theta * rho_diff)) / theta
        else:
            raise NotImplementedError

    if arr.is_discrete():
        log_part = log(prob_b * (1 - exp(theta * rho_diff)))
        return sigma_sum - log_part / theta
    else:
        log_part = log(prob_b * (1 - exp(theta * tau * rho_diff)))
        return tau * arr.rho(theta=p * theta) + sigma_sum - log_part / theta


def delay_prob(arr: Arrival,
               ser: Server,
               theta: float,
               delay_value: int,
               tau=1.0,
               indep=True,
               p=1.0,
               use_standard=True) -> float:
    """Implements stationary bound method"""
    if indep:
        p = 1.0

    stability_check(arr=arr, ser=ser, theta=theta, indep=indep, p=p)
    sigma_sum, rho_diff = get_sigma_rho(arr=arr,
                                        ser=ser,
                                        theta=theta,
                                        indep=indep,
                                        p=p)

    q = get_q(p=p, indep=indep)

    try:
        if not use_standard:
            if arr.is_discrete():
                return exp(
                    -theta * ser.rho(theta=q * theta) * delay_value) * exp(
                        theta * sigma_sum) / (-theta * rho_diff)
            else:
                raise NotImplementedError

        if arr.is_discrete():
            return exp(-theta * ser.rho(theta=q * theta) * delay_value) * exp(
                theta * sigma_sum) / (1 - exp(theta * rho_diff))
        else:
            return exp(-theta * ser.rho(theta=q * theta) * delay_value) * exp(
                theta * (arr.rho(theta=p * theta) * tau + sigma_sum)) / (
                    1 - exp(theta * tau * rho_diff))

    except ZeroDivisionError:
        return inf


def delay(arr: Arrival,
          ser: Server,
          theta: float,
          prob_d: float,
          tau=1.0,
          indep=True,
          p=1.0,
          use_standard=True) -> float:
    """Implements stationary bound method"""
    if indep:
        p = 1.0

    stability_check(arr=arr, ser=ser, theta=theta, indep=indep, p=p)
    sigma_sum, rho_diff = get_sigma_rho(arr=arr,
                                        ser=ser,
                                        theta=theta,
                                        indep=indep,
                                        p=p)

    q = get_q(p=p, indep=indep)

    if not use_standard:
        if arr.is_discrete():
            log_part = log(-prob_d * theta * rho_diff)
            return (sigma_sum - log_part / theta) / ser.rho(theta=q * theta)
        else:
            raise NotImplementedError

    if arr.is_discrete():
        log_part = log(prob_d * (1 - exp(theta * rho_diff)))
        return (sigma_sum - log_part / theta) / ser.rho(theta=q * theta)
    else:
        log_part = log(prob_d * (1 - exp(theta * tau * rho_diff)))
        return (tau * arr.rho(theta=p * theta) + sigma_sum -
                log_part / theta) / ser.rho(theta=q * theta)


def output(arr: Arrival,
           ser: Server,
           theta: float,
           delta_time: int,
           indep=True,
           p=1.0) -> float:
    """Implements stationary bound method"""
    if indep:
        p = 1.0

    stability_check(arr=arr, ser=ser, theta=theta, indep=indep, p=p)
    sigma_sum, rho_diff = get_sigma_rho(arr=arr,
                                        ser=ser,
                                        theta=theta,
                                        indep=indep,
                                        p=p)

    try:
        if arr.is_discrete():
            return exp(theta * arr.rho(theta=p * theta) * delta_time) * exp(
                theta * sigma_sum) / (1 - exp(theta * rho_diff))

        else:
            return exp(theta * arr.rho(theta=p * theta) *
                       (delta_time + 1)) * exp(
                           theta * sigma_sum) / (1 - exp(theta * rho_diff))

    except ZeroDivisionError:
        return inf
