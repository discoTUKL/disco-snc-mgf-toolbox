"""Performance bounds that cannot be described via (sigma, rho)"""

import scipy.optimize

from nc_processes.arrivals_alternative import fbm, regulated_alternative
from nc_processes.service_alternative import constant_rate_alternative

SIGMA_SINGLE = 6.0
RHO_SINGLE = 0.1
N = 5
RATE = 6.0


def mgf_a(theta: float, delta_time: int, process="regulated_alternative"):
    if process == "regulated_alternative":
        return regulated_alternative(
            theta=theta,
            delta_time=delta_time,
            sigma_single=SIGMA_SINGLE,
            rho_single=RHO_SINGLE,
            n=N)

    elif process == "fbm":
        return fbm(
            theta=theta, delta_time=delta_time, lamb=1.0, sigma=1.0, hurst=0.7)

    else:
        raise NameError("{0} is an infeasible arrival process".format(process))


def mgf_s(theta: float, delta_time: int):
    return constant_rate_alternative(
        theta=theta, delta_time=delta_time, rate=RATE)


def delay_prob_alternative(theta: float, delay_value: int, t: int) -> float:
    delay_prob = 0.0

    for _i in range(t + 1):
        delay_prob += mgf_a(
            theta=theta, delta_time=t - _i) * mgf_s(
                theta=theta, delta_time=t + delay_value - _i)

    return delay_prob


def del_prob_alter_opt(delay_value: int, t: int, print_x=False) -> float:
    def helper_fun(theta: float):
        return delay_prob_alternative(
            theta=theta, delay_value=delay_value, t=t)

    grid_res = scipy.optimize.brute(
        func=helper_fun, ranges=(slice(0.05, 15.0, 0.05), ), full_output=True)

    if print_x:
        print("grid search optimal x: ", grid_res[0].tolist())

    return grid_res[1]


def output_alternative(theta: float, s: int, t: int) -> float:
    output = 0.0

    for _i in range(t + 1):
        output += mgf_a(
            theta=theta, delta_time=t - _i) * mgf_s(
                theta=theta, delta_time=s - _i)

    return output
