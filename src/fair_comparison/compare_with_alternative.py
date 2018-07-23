"""Compare with alternative traffic description"""

from math import inf, log, nan

import numpy as np
import scipy.optimize

from library.exceptions import ParameterOutOfBounds
from library.helper_functions import mgf
from library.perform_parameter import PerformParameter
from nc_operations.perform_enum import PerformEnum
from nc_processes.arrivals_alternative import regulated_alternative
from nc_processes.constant_rate_server import ConstantRate
from nc_processes.regulated_arrivals import (LeakyBucketMassOne,
                                             TokenBucketConstant)
from nc_processes.service import Service
from optimization.optimize import Optimize
from single_server.single_server_perform import SingleServerPerform


def delay_prob_leaky(theta: float,
                     delay_value: int,
                     sigma_single: float,
                     rho_single: float,
                     ser: Service,
                     t: int,
                     n: int = 1) -> float:
    if t < 0:
        raise ValueError("sum index t = {0} must be >= 0".format(t))

    sigma_a = n * sigma_single
    rho_a = n * rho_single

    sigma_s = ser.sigma(theta=theta)
    rho_s = ser.rho(theta=theta)

    if rho_a >= -rho_s:
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(rho_a, -rho_s))

    sum_j = 0.0

    # TODO: Look for more bugs

    for _j in range(t):
        try:
            summand = regulated_alternative(
                theta=theta,
                delta_time=_j,
                sigma_single=sigma_single,
                rho_single=rho_single,
                n=n) * mgf(
                    theta=theta, x=_j * rho_s)
        except (FloatingPointError, OverflowError):
            summand = inf

        sum_j += summand
    # print(sum_j)
    # sum_j = 1.0

    try:
        return mgf(
            theta=theta, x=sigma_s + rho_s * delay_value) * (
                mgf(theta=theta, x=sigma_a + (rho_a + rho_s) * t) /
                (1 - mgf(theta=theta, x=rho_a + rho_s)) + sum_j)
    except FloatingPointError:
        return nan


def del_prob_alter_opt(delay_value: int,
                       sigma_single: float,
                       rho_single: float,
                       ser: Service,
                       t: int,
                       n: int = 1,
                       print_x: bool = False) -> float:
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
        except (FloatingPointError, OverflowError):
            return inf

    # np.seterr("raise")
    np.seterr("warn")

    try:
        grid_res = scipy.optimize.brute(
            func=helper_fun,
            ranges=(slice(0.05, 20.0, 0.05), ),
            full_output=True)
    except (FloatingPointError, OverflowError):
        return inf

    if print_x:
        print("grid search optimal x: ", grid_res[0].tolist())

    return grid_res[1]
    # ranges = np.arange(start=0.05, stop=20.0 + 10**(-10), step=0.05).tolist()
    # y_opt = inf
    # theta_opt = 0.0
    #
    # for val in ranges:
    #     candidate_opt = delay_prob_leaky(
    #         theta=val,
    #         delay_value=delay_value,
    #         sigma_single=sigma_single,
    #         rho_single=rho_single,
    #         ser=ser,
    #         t=t,
    #         n=n)
    #     if candidate_opt < y_opt:
    #         y_opt = candidate_opt
    #         theta_opt = val
    #
    # if print_x:
    #     print("grid search optimal x: ", theta_opt)
    #
    # return y_opt


def delay_leaky(theta: float,
                prob_d: float,
                sigma_single: float,
                rho_single: float,
                ser: Service,
                t: int,
                n: int = 1) -> float:
    if t < 0:
        raise ValueError("sum index t = {0} must be >= 0".format(t))

    sigma_a = n * sigma_single
    rho_a = n * rho_single

    sigma_s = ser.sigma(theta=theta)
    rho_s = ser.rho(theta=theta)

    if rho_a >= -rho_s:
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            "the service's rho {1}".format(rho_a, -rho_s))

    sum_j = 0.0

    for _j in range(t):
        try:
            summand = regulated_alternative(
                theta=theta,
                delta_time=_j,
                sigma_single=sigma_single,
                rho_single=rho_single,
                n=n) * mgf(
                    theta=theta, x=_j * rho_s)

        except (FloatingPointError, OverflowError):
            summand = inf

        sum_j += summand

    try:
        return -(log(mgf(theta=theta, x=sigma_s) / prob_d) + log(
            mgf(theta=theta, x=sigma_a + (rho_a + rho_s) * t) /
            (1 - mgf(theta=theta, x=rho_a + rho_s)) + sum_j)) / (theta * rho_s)

    except FloatingPointError:
        return nan


def del_alter_opt(prob_d: float,
                  sigma_single: float,
                  rho_single: float,
                  ser: Service,
                  t: int,
                  n: int = 1,
                  print_x: bool = False) -> float:
    try:

        def helper_fun(theta: float) -> float:
            return delay_leaky(
                theta=theta,
                prob_d=prob_d,
                sigma_single=sigma_single,
                rho_single=rho_single,
                ser=ser,
                t=t,
                n=n)

    except (FloatingPointError, OverflowError):
        return inf

    # np.seterr("raise")
    np.seterr("warn")

    try:
        grid_res = scipy.optimize.brute(
            func=helper_fun,
            ranges=(slice(0.05, 20.0, 0.05), ),
            full_output=True)

    except (FloatingPointError, OverflowError):
        return inf

    if print_x:
        print("grid search optimal x: ", grid_res[0].tolist())

    return grid_res[1]


if __name__ == '__main__':
    DELAY_VAL = 5
    DELAY5 = PerformParameter(
        perform_metric=PerformEnum.DELAY_PROB, value=DELAY_VAL)

    DELAY_PROB_VAL = 10**(-5)
    DELAY_PROB6 = PerformParameter(
        perform_metric=PerformEnum.DELAY, value=DELAY_PROB_VAL)

    NUMBER_AGGREGATIONS = 4

    RHO_SINGLE = 1.0
    SIGMA_SINGLE = 8.0
    SERVICE_RATE = 6.5

    BOUND_LIST = [(0.05, 20.0)]
    DELTA = 0.05
    PRINT_X = True

    CR_SERVER = ConstantRate(SERVICE_RATE)

    TB_CONST = TokenBucketConstant(
        sigma_single=SIGMA_SINGLE,
        rho_single=RHO_SINGLE,
        n=NUMBER_AGGREGATIONS)

    CONST_SINGLE = SingleServerPerform(
        arr=TB_CONST, const_rate=CR_SERVER, perform_param=DELAY5)

    LEAKY_MASS_1 = SingleServerPerform(
        arr=LeakyBucketMassOne(
            sigma_single=SIGMA_SINGLE,
            rho_single=RHO_SINGLE,
            n=NUMBER_AGGREGATIONS),
        const_rate=CR_SERVER,
        perform_param=DELAY5)

    CONST_OPT = Optimize(
        setting=CONST_SINGLE, print_x=PRINT_X).grid_search(
            bound_list=BOUND_LIST, delta=DELTA)
    print("const_opt", CONST_OPT)

    LEAKY_MASS_1_OPT = Optimize(
        setting=LEAKY_MASS_1, print_x=PRINT_X).grid_search(
            bound_list=BOUND_LIST, delta=DELTA)
    print("leaky_mass_1_opt", LEAKY_MASS_1_OPT)

    print("leaky_bucket_alter_opt")
    for _i in range(10):
        leaky_bucket_alter_opt = del_prob_alter_opt(
            delay_value=DELAY_VAL,
            sigma_single=SIGMA_SINGLE,
            rho_single=RHO_SINGLE,
            ser=CR_SERVER,
            t=_i,
            n=NUMBER_AGGREGATIONS,
            print_x=False)
        print("{0} {1}".format(_i, leaky_bucket_alter_opt))

    print("----------------------------------------------")

    CONST_SINGLE2 = SingleServerPerform(
        arr=TB_CONST, const_rate=CR_SERVER, perform_param=DELAY_PROB6)

    LEAKY_MASS_1_2 = SingleServerPerform(
        arr=LeakyBucketMassOne(
            sigma_single=SIGMA_SINGLE,
            rho_single=RHO_SINGLE,
            n=NUMBER_AGGREGATIONS),
        const_rate=CR_SERVER,
        perform_param=DELAY_PROB6)

    CONST_OPT_2 = Optimize(
        setting=CONST_SINGLE2, print_x=PRINT_X).grid_search(
            bound_list=BOUND_LIST, delta=DELTA)
    print("const_opt_2", CONST_OPT_2)

    LEAKY_MASS_1_OPT_2 = Optimize(
        setting=LEAKY_MASS_1_2, print_x=PRINT_X).grid_search(
            bound_list=BOUND_LIST, delta=DELTA)
    print("leaky_mass_1_opt_2", LEAKY_MASS_1_OPT_2)

    print("leaky_bucket_alter_opt_2")
    for _i in range(10):
        leaky_bucket_alter_opt_2 = del_alter_opt(
            prob_d=DELAY_PROB_VAL,
            sigma_single=SIGMA_SINGLE,
            rho_single=RHO_SINGLE,
            ser=CR_SERVER,
            t=_i,
            n=NUMBER_AGGREGATIONS,
            print_x=False)
        print("{0} {1}".format(_i, leaky_bucket_alter_opt_2))
