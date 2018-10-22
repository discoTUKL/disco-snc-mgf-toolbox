"""Numerical evaluation of the Input-bound"""

from math import inf
from typing import List

import numpy as np
import scipy.optimize

from library.exceptions import ParameterOutOfBounds
from library.helper_functions import mgf
from nc_operations.operations import Convolve, Deconvolve, Leftover
from nc_operations.performance_bounds import delay_prob
from nc_processes.arrival import Arrival
from nc_processes.arrival_distribution import DM1, ArrivalDistribution
from nc_processes.constant_rate_server import ConstantRate
from nc_processes.service import Service


def new_delay(theta: float, tt: int, delay: int, a_1: ArrivalDistribution,
              a_2: ArrivalDistribution, a_3: ArrivalDistribution,
              s_1: ConstantRate, s_3: ConstantRate) -> float:
    sigmas = a_1.sigma(theta=theta) + a_2.sigma(theta=theta) + a_3.sigma(
        theta=theta) + s_1.sigma(theta=theta) + s_3.sigma(theta=theta)

    outer = 0.0
    for k in range(tt + 1):
        inner = 0.0
        for i in range(k, tt + delay + 1):
            inner += mgf(
                theta=theta,
                x=s_1.rho(theta=theta) * (i - k) + a_2.rho(theta=theta) *
                (i - k) + s_3.rho(theta=theta) * (tt - i))

        outer += mgf(theta=theta, x=a_1.rho(theta=theta) * (tt - k)) * inner

    return mgf(
        theta=theta,
        x=a_3.rho(theta=theta) *
        (tt + delay) + s_3.rho(theta=theta) * delay + sigmas) * outer


def new_delay_opt(tt: int,
                  delay: int,
                  a_1: ArrivalDistribution,
                  a_2: ArrivalDistribution,
                  a_3: ArrivalDistribution,
                  s_1: ConstantRate,
                  s_3: ConstantRate,
                  print_x=False) -> float:
    def helper_fun(param: float) -> float:
        try:
            return new_delay(
                theta=param,
                tt=tt,
                delay=delay,
                a_1=a_1,
                a_2=a_2,
                a_3=a_3,
                s_1=s_1,
                s_3=s_3)
        except (FloatingPointError, OverflowError, ParameterOutOfBounds):
            return inf

    # np.seterr("raise")
    np.seterr("warn")

    try:
        grid_res = scipy.optimize.brute(
            func=helper_fun, ranges=[slice(0.05, 4.0, 0.05)], full_output=True)
    except (FloatingPointError, OverflowError):
        return inf

    if print_x:
        print("grid search optimal parameter: theta={0}".format(
            grid_res[0].tolist()[0]))

    return grid_res[1]


def standard_delay(theta: float, p: float, delay: int,
                   a_1: ArrivalDistribution, a_2: ArrivalDistribution,
                   a_3: ArrivalDistribution, s_1: ConstantRate,
                   s_2: ConstantRate, s_3: ConstantRate) -> float:
    f_3_output: Arrival = Deconvolve(
        arr=a_3, ser=Leftover(arr=Deconvolve(arr=a_2, ser=s_1), ser=s_2))
    s_net: Service = Convolve(
        ser1=Leftover(arr=a_2, ser=s_1),
        ser2=Leftover(arr=f_3_output, ser=s_3),
        indep=False,
        p=p)

    return delay_prob(
        arr=a_1, ser=s_net, theta=theta, delay_value=delay, indep=True)


def standard_delay_opt(delay: int,
                       a_1: ArrivalDistribution,
                       a_2: ArrivalDistribution,
                       a_3: ArrivalDistribution,
                       s_1: ConstantRate,
                       s_2: ConstantRate,
                       s_3: ConstantRate,
                       print_x=False) -> float:
    def helper_fun(param_list: List[float]) -> float:
        try:
            return standard_delay(
                theta=param_list[0],
                delay=delay,
                a_1=a_1,
                a_2=a_2,
                a_3=a_3,
                s_1=s_1,
                s_2=s_2,
                s_3=s_3,
                p=param_list[1])
        except (FloatingPointError, OverflowError, ParameterOutOfBounds):
            return inf

    # np.seterr("raise")
    np.seterr("warn")

    try:
        grid_res = scipy.optimize.brute(
            func=helper_fun,
            ranges=(slice(0.05, 4.0, 0.05), slice(1.05, 10.0, 0.05)),
            full_output=True)
    except (FloatingPointError, OverflowError):
        return inf

    if print_x:
        print("grid search optimal parameter: theta={0}, p={1}".format(
            grid_res[0].tolist()[0], grid_res[0].tolist()[1]))

    return grid_res[1]


if __name__ == '__main__':
    TT = 20
    DELAY = 5
    A_1 = DM1(lamb=1.0, n=1)
    A_2 = DM1(lamb=1.0, n=1)
    A_3 = DM1(lamb=1.0, n=1)
    S_1 = ConstantRate(rate=6.0)
    S_2 = ConstantRate(rate=6.0)
    S_3 = ConstantRate(rate=6.0)

    print(
        "new bound",
        new_delay_opt(
            tt=TT,
            delay=DELAY,
            a_1=A_1,
            a_2=A_2,
            a_3=A_3,
            s_1=S_1,
            s_3=S_3,
            print_x=False))

    print(
        "standard bound",
        standard_delay_opt(
            delay=DELAY,
            a_1=A_1,
            a_2=A_2,
            a_3=A_3,
            s_1=S_1,
            s_2=S_2,
            s_3=S_3,
            print_x=False))
