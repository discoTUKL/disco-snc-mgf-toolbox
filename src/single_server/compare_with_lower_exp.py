"""Compare with alternative traffic description"""

import csv
from math import exp, inf, log, nan

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
from optimization.optimize_new import OptimizeNew
from single_server.single_server_perform import SingleServerPerform
from library.monte_carlo_dist import MonteCarloDist
from library.mc_enum import MCEnum
from nc_processes.arrival_enum import ArrivalEnum
from tqdm import tqdm
from library.array_to_results import three_col_array_to_results


def output_lower_exp_dm1(theta: float, s: int, t: int, lamb: float,
                         rate: float, a: float) -> float:
    if t < s:
        raise ValueError("sum index t = {0} must be >= s={1}".format(t, s))

    if a <= 1:
        raise ParameterOutOfBounds("base a={0} must be >0".format(a))

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
            ranges=(slice(0.05, 10.0, 0.05), slice(1.05, 5.0, 0.05)),
            full_output=True)
    except (FloatingPointError, OverflowError):
        return inf

    if print_x:
        print("grid search optimal parameter: theta={0}, a={1}".format(
            grid_res[0].tolist()[0], grid_res[0].tolist()[1]))

    return grid_res[1]


def csv_single_param_power(start_time: int, delta_time: int,
                           mc_dist: MonteCarloDist) -> dict:
    total_iterations = 10**2
    metric = "relative"

    delta = 0.05

    size_array = [total_iterations, 2]
    # [rows, columns]

    if mc_dist.mc_enum == MCEnum.UNIFORM:
        param_array = np.random.uniform(
            low=0, high=mc_dist.param_list[0], size=size_array)
    elif mc_dist.mc_enum == MCEnum.EXPONENTIAL:
        param_array = np.random.exponential(
            scale=mc_dist.param_list[0], size=size_array)
    else:
        raise ValueError("Distribution parameter {0} is infeasible".format(
            mc_dist.mc_enum))

    res_array = np.empty([total_iterations, 3])

    for i in tqdm(range(total_iterations)):
        setting = SingleServerPerform(
            arr=DM1(lamb=param_array[i, 0]),
            const_rate=ConstantRate(rate=param_array[i, 1]),
            perform_param=PerformParameter(
                perform_metric=PerformEnum.OUTPUT, value=delta_time))

        theta_bounds = [(0.1, 4.0)]
        bound_array = theta_bounds[:]

        res_array[i, 0] = Optimize(setting=setting).grid_search(
            bound_list=bound_array, delta=delta)

        bound_array_power = theta_bounds[:]
        bound_array_power.append((0.9, 4.0))

        res_array[i, 1] = OptimizeNew(setting_new=setting).grid_search(
            bound_list=bound_array_power, delta=delta)

        res_array[i, 2] = output_lower_exp_dm1_opt(
            s=start_time,
            t=start_time + delta_time,
            lamb=param_array[i, 0],
            rate=param_array[i, 1])

        if (res_array[i, 0] == inf or res_array[i, 1] == inf
                or res_array[i, 2] == inf or res_array[i, 0] == nan
                or res_array[i, 1] == nan or res_array[i, 2] == nan):
            res_array[i, ] = nan

    res_dict = three_col_array_to_results(
        arrival_enum=ArrivalEnum.DM1, metric=metric, res_array=res_array)

    res_dict.update({
        "delta_time": delta_time,
        "optimization": "grid_search",
        "metric": "relative",
        "iterations": total_iterations,
        "MCDistribution": mc_dist.to_name(),
        "MCParam": mc_dist.param_to_string()
    })

    with open(
            "single_output_DM1_results_MC{0}_power_exp.csv".format(
                mc_dist.to_name()), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in res_dict.items():
            writer.writerow([key, value])

    return res_dict


if __name__ == '__main__':
    S = 30
    DELTA_TIME = 10

    OUTPUT5 = PerformParameter(
        perform_metric=PerformEnum.OUTPUT, value=DELTA_TIME)

    LAMB = 1.0
    SERVICE_RATE = 1.1

    BOUND_LIST = [(0.05, 10.0)]
    BOUND_LIST_NEW = [(0.05, 10.0), (1.05, 20.0)]
    DELTA = 0.05
    PRINT_X = False

    CR_SERVER = ConstantRate(SERVICE_RATE)

    EXP_ARRIVAL = DM1(lamb=LAMB)

    DM1_SINGLE = SingleServerPerform(
        arr=EXP_ARRIVAL, const_rate=CR_SERVER, perform_param=OUTPUT5)

    DM1_STANDARD_OPT = Optimize(
        setting=DM1_SINGLE, print_x=PRINT_X).grid_search(
            bound_list=BOUND_LIST, delta=DELTA)
    print("DM1 Standard Opt: ", DM1_STANDARD_OPT)

    DM1_POWER_OPT = OptimizeNew(
        setting_new=DM1_SINGLE, print_x=PRINT_X).grid_search(
            bound_list=BOUND_LIST_NEW, delta=DELTA)
    print("DM1 Power Opt: ", DM1_POWER_OPT)

    DM1_EXP_OPT = output_lower_exp_dm1_opt(
        s=S, t=S + DELTA_TIME, lamb=LAMB, rate=SERVICE_RATE, print_x=PRINT_X)
    print("DM1 Exp Opt: ", DM1_EXP_OPT)
