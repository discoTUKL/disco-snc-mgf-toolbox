"""Compare with alternative traffic description"""

from math import exp, inf

import scipy.optimize

from library.perform_parameter import PerformParameter
from nc_operations.perform_enum import PerformEnum
from nc_processes.arrivals_alternative import regulated_alternative
from nc_processes.regulated_arrivals import (LeakyBucketMassOne,
                                             TokenBucketConstant)
from nc_processes.service import Service
from nc_processes.constant_rate_server import ConstantRate
from optimization.optimize import Optimize
from single_server.single_server_perform import SingleServerPerform


def delay_prob_leaky(theta: float,
                     delay_value: int,
                     sigma_single: float,
                     rho_single: float,
                     ser: Service,
                     t: int,
                     n=1) -> float:
    if t < 0:
        raise ValueError("sum index t = {0} must be >= 0".format(t))

    sigma_s = ser.sigma(theta=theta)
    rho_s = ser.rho(theta=theta)

    sum_j = 0.0

    # TODO: Look for more bugs

    for _j in range(t):
        sum_j += regulated_alternative(
            theta=theta,
            delta_time=_j,
            sigma_single=sigma_single,
            rho_single=rho_single,
            n=n) * exp(_j * theta * rho_s)
    # print(sum_j)
    # sum_j = 1.0

    return exp(theta * (sigma_s + rho_s * delay_value)) * (
        exp(theta * (n * (sigma_single + rho_single * t) + rho_s * t)) /
        (1 - exp(theta * (n * rho_single + rho_s))) + sum_j)


def del_prob_alter_opt(delay_value: int,
                       sigma_single: float,
                       rho_single: float,
                       ser: Service,
                       t: int,
                       n=1,
                       print_x=False) -> float:
    def helper_fun(theta: float) -> float:
        try:
            return delay_prob_leaky(
                theta=theta,
                delay_value=delay_value,
                sigma_single=sigma_single,
                rho_single=rho_single,
                ser=ser,
                t=t,
                n=n)
        except OverflowError:
            return inf

    grid_res = scipy.optimize.brute(
        func=helper_fun, ranges=(slice(0.05, 15.0, 0.05), ), full_output=True)

    if print_x:
        print("grid search optimal x: ", grid_res[0].tolist())

    return grid_res[1]


if __name__ == '__main__':
    DELAY_VAL = 5
    DELAYPROB5 = PerformParameter(
        perform_metric=PerformEnum.DELAY_PROB, value=DELAY_VAL)

    NUMBER_AGGREGATIONS = 5

    RHO_SINGLE = 0.1
    SIGMA_SINGLE = 6.5
    SERVICE_RATE = 6.0

    BOUND_LIST = [(0.05, 15.0)]
    DELTA = 0.05
    PRINT_X = True

    constant_rate_server = ConstantRate(SERVICE_RATE)

    tb_const = TokenBucketConstant(
        sigma_single=SIGMA_SINGLE,
        rho_single=RHO_SINGLE,
        n=NUMBER_AGGREGATIONS)

    const_single = SingleServerPerform(
        arr=tb_const,
        const_rate=constant_rate_server,
        perform_param=DELAYPROB5)

    leaky_mass_1 = SingleServerPerform(
        arr=LeakyBucketMassOne(
            sigma_single=SIGMA_SINGLE,
            rho_single=RHO_SINGLE,
            n=NUMBER_AGGREGATIONS),
        const_rate=constant_rate_server,
        perform_param=DELAYPROB5)

    const_opt = Optimize(
        setting=const_single, print_x=PRINT_X).grid_search(
            bound_list=BOUND_LIST, delta=DELTA)
    print("const_opt", const_opt)

    leaky_mass_1_opt = Optimize(
        setting=leaky_mass_1, print_x=PRINT_X).grid_search(
            bound_list=BOUND_LIST, delta=DELTA)
    print("leaky_mass_1_opt", leaky_mass_1_opt)

    leaky_bucket_alter_opt = del_prob_alter_opt(
        delay_value=DELAY_VAL,
        sigma_single=SIGMA_SINGLE,
        rho_single=RHO_SINGLE,
        ser=constant_rate_server,
        t=1,
        n=NUMBER_AGGREGATIONS,
        print_x=PRINT_X)
    print("leaky_bucket_alter_opt", leaky_bucket_alter_opt)
