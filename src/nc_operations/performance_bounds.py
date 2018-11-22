"""Performance bounds"""

from math import inf, log

from library.exceptions import ParameterOutOfBounds
from library.helper_functions import is_equal, get_q, mgf
from nc_processes.arrival import Arrival
from nc_processes.service import Service


def backlog_prob(arr: Arrival,
                 ser: Service,
                 theta: float,
                 backlog_value: float,
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
            f"The arrivals' rho {rho_a_p} has to be smaller than"
            f"the service's rho {-rho_s_q}")

    try:
        return mgf(
            theta=theta, x=-backlog_value + sigma_a_p + sigma_s_q) / (
                1 - mgf(theta=theta, x=rho_a_p + rho_s_q))
    except ZeroDivisionError:
        return inf


def backlog_prob_t(arr: Arrival,
                   ser: Service,
                   theta: float,
                   tt: int,
                   backlog_value: float,
                   indep=True,
                   p=1.0) -> float:
    """Implements time dependent method"""
    if indep:
        p = 1.0

    q = get_q(p=p, indep=indep)

    rho_a_p = arr.rho(theta=p * theta)
    sigma_a_p = arr.sigma(theta=p * theta)
    rho_s_q = ser.rho(theta=q * theta)
    sigma_s_q = ser.sigma(theta=q * theta)

    if is_equal(rho_a_p, -rho_s_q):
        return mgf(
            theta=theta, x=-backlog_value + sigma_a_p + sigma_s_q) * (tt + 1)

    elif rho_a_p > -rho_s_q:
        return mgf(
            theta=theta,
            x=-backlog_value + rho_a_p + rho_s_q * tt + sigma_a_p +
            sigma_s_q) / (1 - mgf(theta=theta, x=-(rho_a_p + rho_s_q)))

    else:
        return backlog_prob(
            arr=arr,
            ser=ser,
            theta=theta,
            backlog_value=backlog_value,
            indep=indep,
            p=p)


def backlog(arr: Arrival,
            ser: Service,
            theta: float,
            prob_b: float,
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
            f"The arrivals' rho {rho_a_p} has to be smaller than"
            f"the service's rho {-rho_s_q}")

    log_part = log(prob_b * (1 - mgf(theta=theta, x=rho_a_p + rho_s_q)))

    return sigma_a_p + sigma_s_q - log_part / theta


def backlog_t(arr: Arrival,
              ser: Service,
              theta: float,
              prob_b: float,
              tt: int,
              indep=True,
              p=1.0) -> float:
    """Implements time dependent method"""
    if indep:
        p = 1.0

    q = get_q(p=p, indep=indep)

    rho_a_p = arr.rho(theta=p * theta)
    sigma_a_p = arr.sigma(theta=p * theta)
    rho_s_q = ser.rho(theta=q * theta)
    sigma_s_q = ser.sigma(theta=q * theta)

    if is_equal(rho_a_p, -rho_s_q):
        log_part = log(prob_b / (tt + 1))

        return sigma_a_p + sigma_s_q - log_part / theta

    elif rho_a_p > -rho_s_q:
        log_part = log(prob_b * (1 - mgf(theta=theta, x=-(rho_a_p + rho_s_q))))

        return rho_a_p + rho_s_q * tt + sigma_a_p + sigma_s_q - log_part / theta

    else:
        return backlog(
            arr=arr, ser=ser, theta=theta, prob_b=prob_b, indep=indep, p=p)


def delay_prob(arr: Arrival,
               ser: Service,
               theta: float,
               delay_value: int,
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
            f"The arrivals' rho {rho_a_p} has to be smaller than"
            f"the service's rho {-rho_s_q}")

    try:
        return mgf(
            theta=theta, x=sigma_a_p + sigma_s_q + rho_s_q * delay_value) / (
                1 - mgf(theta=theta, x=rho_a_p + rho_s_q))
    except ZeroDivisionError:
        return inf


def delay_prob_t(arr: Arrival,
                 ser: Service,
                 theta: float,
                 tt: int,
                 delay_value: int,
                 indep=True,
                 p=1.0) -> float:
    """Implements time dependent method"""
    if indep:
        p = 1.0

    q = get_q(p=p, indep=indep)

    rho_a_p = arr.rho(theta=p * theta)
    sigma_a_p = arr.sigma(theta=p * theta)
    rho_s_q = ser.rho(theta=q * theta)
    sigma_s_q = ser.sigma(theta=q * theta)

    if is_equal(rho_a_p, -rho_s_q):
        return mgf(
            theta=theta,
            x=rho_s_q * delay_value + sigma_a_p + sigma_s_q) * (tt + 1)

    elif rho_a_p > -rho_s_q:
        return mgf(
            theta=theta,
            x=rho_a_p * tt + rho_s_q * (tt + delay_value) + sigma_a_p +
            sigma_s_q) / (1 - mgf(theta=theta, x=-(rho_a_p + rho_s_q)))

    else:
        return delay_prob(
            arr=arr,
            ser=ser,
            theta=theta,
            delay_value=delay_value,
            indep=indep,
            p=p)


def delay(arr: Arrival,
          ser: Service,
          theta: float,
          prob_d: float,
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
            f"The arrivals' rho {rho_a_p} has to be smaller than"
            f"the service's rho {-rho_s_q}")

    log_part = log(prob_d * (1 - mgf(theta=theta, x=rho_a_p + rho_s_q)))

    return (log_part / theta - (sigma_a_p + sigma_s_q)) / rho_s_q


def delay_t(arr: Arrival,
            ser: Service,
            theta: float,
            prob_d: float,
            tt: int,
            indep=True,
            p=1.0) -> float:
    """Implements time dependent method"""
    if indep:
        p = 1.0

    q = get_q(p=p, indep=indep)

    rho_a_p = arr.rho(theta=p * theta)
    sigma_a_p = arr.sigma(theta=p * theta)
    rho_s_q = ser.rho(theta=q * theta)
    sigma_s_q = ser.sigma(theta=q * theta)

    if rho_a_p == -rho_s_q:
        log_part = log(prob_d / (tt + 1))

        return (log_part / theta - (sigma_a_p + sigma_s_q)) / rho_s_q

    elif rho_a_p > -rho_s_q:
        log_part = log(prob_d * (1 - mgf(theta=theta, x=-(rho_a_p + rho_s_q))))

        return log_part / theta - (
            rho_a_p + rho_s_q * tt + sigma_a_p + sigma_s_q) / rho_s_q

    else:
        return delay(
            arr=arr, ser=ser, theta=theta, prob_d=prob_d, indep=indep, p=p)


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

    if rho_a_p >= -rho_s_q:
        raise ParameterOutOfBounds(
            f"The arrivals' rho {rho_a_p} has to be smaller than"
            f"the service's rho {-rho_s_q}")

    try:
        return mgf(
            theta=theta, x=rho_a_p * delta_time + sigma_a_p + sigma_s_q) / (
                1 - mgf(theta=theta, x=rho_a_p + rho_s_q))

    except ZeroDivisionError:
        return inf


def output_t(arr: Arrival,
             ser: Service,
             theta: float,
             tt: int,
             ss: int,
             indep=True,
             p=1.0) -> float:
    """Implements time dependent method"""
    if indep:
        p = 1.0

    q = get_q(p=p, indep=indep)

    rho_a_p = arr.rho(theta=p * theta)
    sigma_a_p = arr.sigma(theta=p * theta)
    rho_s_q = ser.rho(theta=q * theta)
    sigma_s_q = ser.sigma(theta=q * theta)

    if is_equal(rho_a_p, -rho_s_q):
        return mgf(
            theta=theta, x=rho_a_p *
            (tt - ss) + sigma_a_p + sigma_s_q) * (ss + 1)

    elif rho_a_p > -rho_s_q:
        return mgf(
            theta=theta, x=rho_a_p * tt + rho_s_q * ss + sigma_a_p +
            sigma_s_q) / (1 - mgf(theta=theta, x=-(rho_a_p + rho_s_q)))

    else:
        return output(
            arr=arr,
            ser=ser,
            theta=theta,
            delta_time=tt - ss,
            indep=indep,
            p=p)
