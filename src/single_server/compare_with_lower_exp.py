"""Compare with alternative traffic description"""

from math import exp, inf, log

import numpy as np
import scipy.optimize

from typing import List

from library.exceptions import ParameterOutOfBounds
from library.perform_parameter import PerformParameter
from nc_operations.perform_enum import PerformEnum
from nc_processes.arrivals_alternative import expect_dm1, long_term_dm1
from nc_processes.arrival_distribution import DM1
from nc_processes.constant_rate_server import ConstantRate
from nc_processes.service_alternative import (expect_const_rate,
                                              long_term_const_rate)
from optimization.optimize import Optimize
from single_server.single_server_perform import SingleServerPerform


def output_lower_exp_dm1(theta: float, s: int, t: int, lamb: float,
                         rate: float, a: float) -> float:
    if t < s:
        raise ValueError("sum index t = {0} must be >= s={1}".format(t, s))

    if a <= 1:
        raise ValueError("base a={0} must be >0".format(a))

    if long_term_dm1(lamb=lamb) >= long_term_const_rate(rate=rate):
        raise ParameterOutOfBounds(
            ("The arrivals' long term rate has to be smaller than"
             "the service's long term rate {1}").format(
                 long_term_dm1(lamb=lamb), long_term_const_rate(rate=rate)))

    sum_j = 0.0

    for _i in range(s + 1):
        try:
            exponent = expect_dm1(
                theta=theta, delta_time=t - _i, lamb=lamb) - expect_const_rate(
                    theta=theta, delta_time=s - _i, rate=rate)
            summand = a**exp(exponent)
        except (FloatingPointError, OverflowError):
            summand = inf

        sum_j += summand

    return log(sum_j) / log(a)


def output_lower_exp_dm1_opt(s: int,
                             t: int,
                             lamb: float,
                             rate: float,
                             print_x: bool = False) -> float:
    def helper_fun(param_list: List[float]) -> float:
        try:
            return output_lower_exp_dm1(
                theta=param_list[0],
                s=s,
                t=t,
                lamb=lamb,
                rate=rate,
                a=param_list[1])
        except (FloatingPointError, OverflowError, ParameterOutOfBounds):
            return inf

    # np.seterr("raise")
    np.seterr("warn")

    try:
        grid_res = scipy.optimize.brute(
            func=helper_fun,
            ranges=(slice(0.05, 5.0, 0.05), slice(1.05, 3.0, 0.05)),
            full_output=True)
    except (FloatingPointError, OverflowError):
        return inf

    if print_x:
        print("grid search optimal x: ", grid_res[0].tolist())

    return grid_res[1]


if __name__ == '__main__':
    S = 2
    T = 7
    DELTA_TIME = T - S

    OUTPUT5 = PerformParameter(
        perform_metric=PerformEnum.OUTPUT, value=DELTA_TIME)

    LAMB = 1.0
    SERVICE_RATE = 2.0

    BOUND_LIST = [(1.05, 5.0)]
    DELTA = 0.05
    PRINT_X = True

    CR_SERVER = ConstantRate(SERVICE_RATE)

    EXP_ARRIVAL = DM1(lamb=LAMB)

    DM1_STANDARD = SingleServerPerform(
        arr=EXP_ARRIVAL, const_rate=CR_SERVER, perform_param=OUTPUT5)

    DM1_STANDARD_OPT = Optimize(
        setting=DM1_STANDARD, print_x=PRINT_X).grid_search(
            bound_list=BOUND_LIST, delta=DELTA)
    print("DM1 Standard Opt: ", DM1_STANDARD_OPT)

    DM1_EXP_OPT = output_lower_exp_dm1_opt(
        s=S, t=T, lamb=LAMB, rate=SERVICE_RATE, print_x=True)
    print("DM1 Exp Opt: ", DM1_EXP_OPT)
