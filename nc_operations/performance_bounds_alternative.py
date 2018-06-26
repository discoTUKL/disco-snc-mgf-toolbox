"""Class for all performance_bounds that cannot be described via (sigma, rho)"""

from nc_processes.arrivals_alternative import regulated_alternative
from nc_processes.service_alternative import constant_rate_alternative


def mgf_a(theta: float, delta_time: int, process="regulated_alternative"):
    if process == "regulated_alternative":
        return regulated_alternative(
            theta=theta,
            delta_time=delta_time,
            sigma_single=1.0,
            rho_single=1.0,
            n=1)
    else:
        raise NameError("{0} is an infeasible arrival process".format(process))


def mgf_s(theta: float, delta_time: int):
    return constant_rate_alternative(
        theta=theta, delta_time=delta_time, rate=2.0)


def delay_prob_alternative(theta: float, delay_value: int, t: int) -> float:
    delay_prob = 0.0

    for _i in range(t + 1):
        delay_prob += mgf_a(
            theta=theta, delta_time=t - _i) * mgf_s(
                theta=theta, delta_time=t + delay_value - _i)

    return delay_prob


def output_alternative(theta: float, s: int, t: int) -> float:
    output = 0.0

    for _i in range(t + 1):
        output += mgf_a(
            theta=theta, delta_time=t - _i) * mgf_s(
                theta=theta, delta_time=s - _i)

    return output
