"""Compare with alternative traffic description"""

from math import exp, inf

import scipy.optimize

from library.perform_parameter import PerformParameter
from nc_operations.perform_metric import PerformMetric
from nc_processes.arrivals_alternative import regulated_alternative
from nc_processes.regulated_arrivals import (LeakyBucketMassOne,
                                             TokenBucketConstant)
from nc_processes.service_alternative import constant_rate_alternative
from nc_processes.constant_rate_server import ConstantRate
from optimization.optimize import Optimize
from single_server.single_server_perform import SingleServerPerform


def delay_prob_leaky(theta: float,
                     delay_value: int,
                     sigma_single: float,
                     rho_single: float,
                     rate: float,
                     t: int,
                     n=1) -> float:
    delay_prob = 0.0

    for i in range(t + 1):
        delay_prob += regulated_alternative(
            theta=theta,
            delta_time=t - i,
            sigma_single=sigma_single,
            rho_single=rho_single,
            n=n) * constant_rate_alternative(
                theta=theta, delta_time=t + delay_value - i, rate=rate)

    return delay_prob


def del_prob_alter_opt(delay_value: int,
                       sigma_single: float,
                       rho_single: float,
                       rate: float,
                       t: int,
                       n=1,
                       print_x=False) -> float:
    def helper_fun(theta: float) -> float:
        return delay_prob_leaky(
            theta=theta,
            delay_value=delay_value,
            sigma_single=sigma_single,
            rho_single=rho_single,
            rate=rate,
            t=t,
            n=n)

    grid_res = scipy.optimize.brute(
        func=helper_fun, ranges=(slice(0.05, 15.0, 0.05), ), full_output=True)

    if print_x:
        print("grid search optimal x: ", grid_res[0].tolist())

    return grid_res[1]


def delay_prob_leaky_approx(theta: float,
                            delay_value: int,
                            sigma_single: float,
                            rho_single: float,
                            rate: float,
                            n=1) -> float:
    try:
        one_minus_rate = 1 / (1 - exp(-theta * rate))
        sigma_rho_frac = rho_single / (sigma_single + rho_single)

        return exp(-theta * rate * delay_value) * (
            one_minus_rate + sigma_rho_frac *
            (exp(theta * n * sigma_single) /
             (1 - exp(theta * (n * rho_single - rate))) - one_minus_rate))

    except ZeroDivisionError:
        return inf


def delay_prob_leaky_approx_opt(delay_value: int,
                                sigma_single: float,
                                rho_single: float,
                                rate: float,
                                n=1,
                                print_x=False) -> float:
    def helper_fun(theta: float):
        return delay_prob_leaky_approx(
            theta=theta,
            delay_value=delay_value,
            sigma_single=sigma_single,
            rho_single=rho_single,
            rate=rate,
            n=n)

    grid_res = scipy.optimize.brute(
        func=helper_fun, ranges=(slice(0.05, 15.0, 0.05), ), full_output=True)

    if print_x:
        print("grid search optimal x: ", grid_res[0].tolist())

    return grid_res[1]


if __name__ == '__main__':
    DELAY_VAL = 5
    DELAYPROB5 = PerformParameter(
        perform_metric=PerformMetric.DELAY_PROB, value=DELAY_VAL)

    NUMBER_AGGREGATIONS = 5

    RHO_SINGLE = 0.1
    SERVICE_RATE = 6.0
    SIGMA_SINGLE = 6.0

    BOUND_LIST = [(0.05, 15.0)]
    DELTA = 0.05
    PRINT_X = True

    constant_rate_server = ConstantRate(SERVICE_RATE)

    tb_const = TokenBucketConstant(
        sigma_single=SIGMA_SINGLE,
        rho_single=RHO_SINGLE,
        n=NUMBER_AGGREGATIONS)

    const_single = SingleServerPerform(
        arr=tb_const, const_rate=constant_rate_server, perform_param=DELAYPROB5)

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

    # delay_value: int, sigma_single: float,
    #                        rho_single: float, rate: float, t: int, n=1,
    #                        print_x=False

    leaky_bucket_alter_opt = del_prob_alter_opt(
        delay_value=DELAY_VAL,
        sigma_single=SIGMA_SINGLE,
        rho_single=RHO_SINGLE,
        rate=SERVICE_RATE,
        t=30,
        n=NUMBER_AGGREGATIONS,
        print_x=PRINT_X)
    print("leaky_bucket_alter_opt", leaky_bucket_alter_opt)

    leaky_bucket_alter_approx_opt = delay_prob_leaky_approx_opt(
        delay_value=DELAY_VAL,
        sigma_single=SIGMA_SINGLE,
        rho_single=RHO_SINGLE,
        rate=SERVICE_RATE,
        n=NUMBER_AGGREGATIONS,
        print_x=PRINT_X)
    print("leaky_bucket_alter_approx_opt", leaky_bucket_alter_approx_opt)
