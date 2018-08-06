"""Compare with alternative traffic description"""

import csv
from math import exp, inf, log, nan
from multiprocessing import Process
from typing import List

import numpy as np
import scipy.optimize
from tqdm import tqdm

from library.array_to_results import three_col_array_to_results
from library.exceptions import ParameterOutOfBounds
from library.mc_enum import MCEnum
from library.monte_carlo_dist import MonteCarloDist
from library.perform_parameter import PerformParameter
from nc_operations.perform_enum import PerformEnum
from nc_processes.arrival_distribution import DM1
from nc_processes.arrival_enum import ArrivalEnum
from nc_processes.arrivals_alternative import expect_dm1
from nc_processes.constant_rate_server import ConstantRate
from nc_processes.service_alternative import expect_const_rate
from optimization.optimize import Optimize
from optimization.optimize_new import OptimizeNew
from single_server.single_server_perform import SingleServerPerform


def f_exp(theta: float, i: int, s: int, t: int, lamb: float, rate: float,
          a: float) -> float:
    return a**(exp(theta * expect_dm1(delta_time=t - i, lamb=lamb) -
                   expect_const_rate(delta_time=s - i, rate=rate)))


def f_double_prime(theta: float, i: int, s: int, t: int, lamb: float,
                   rate: float, a: float) -> float:
    help_1 = exp(
        theta * (expect_dm1(delta_time=t - i, lamb=lamb) - rate * (s - i)))
    return (theta**2) * log(a) * help_1 * (a**help_1) * (log(a) * help_1 + 1)


def output_lower_exp_dm1(theta: float, s: int, delta_time: int, lamb: float,
                         rate: float, a: float) -> float:
    if 1 / lamb >= rate:
        raise ParameterOutOfBounds(
            ("The arrivals' long term rate {0} has to be smaller than"
             "the service's long term rate {1}").format(1 / lamb, rate))

    if theta <= 0:
        raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

    if a <= 1:
        raise ParameterOutOfBounds("base a={0} must be >0".format(a))

    sum_j = 0.0

    for _i in range(s + 1):
        try:
            summand = f_exp(
                theta=theta,
                i=_i,
                s=s,
                t=s + delta_time,
                lamb=lamb,
                rate=rate,
                a=a)
        except (FloatingPointError, OverflowError):
            summand = inf

        sum_j += summand

    return log(sum_j) / log(a)


def output_lower_exp_dm1_opt(s: int,
                             delta_time: int,
                             lamb: float,
                             rate: float,
                             print_x: bool = False) -> float:
    def helper_fun(param_list: List[float]) -> float:
        try:
            return output_lower_exp_dm1(
                theta=param_list[0],
                s=s,
                delta_time=delta_time,
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
            ranges=(slice(0.05, 4.0, 0.05), slice(1.05, 10.0, 0.05)),
            full_output=True)
    except (FloatingPointError, OverflowError):
        return inf

    if print_x:
        print("grid search optimal parameter: theta={0}, a={1}".format(
            grid_res[0].tolist()[0], grid_res[0].tolist()[1]))

    return grid_res[1]


def delay_prob_lower_exp_dm1(theta: float, t: int, delay: int, lamb: float,
                             rate: float, a: float) -> float:
    if 1 / lamb >= rate:
        raise ParameterOutOfBounds(
            ("The arrivals' long term rate {0} has to be smaller than"
             "the service's long term rate {1}").format(1 / lamb, rate))

    if theta <= 0:
        raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

    if a <= 1:
        raise ParameterOutOfBounds("base a={0} must be >0".format(a))

    sum_j = 0.0

    for _i in range(t + 1):
        try:
            summand = f_exp(
                theta=theta, i=_i, s=t + delay, t=t, lamb=lamb, rate=rate, a=a)
        except (FloatingPointError, OverflowError):
            summand = inf

        sum_j += summand

    return log(sum_j) / log(a)


def delay_prob_lower_exp_dm1_opt(t: int,
                                 delay: int,
                                 lamb: float,
                                 rate: float,
                                 print_x: bool = False) -> float:
    def helper_fun(param_list: List[float]) -> float:
        try:
            return delay_prob_lower_exp_dm1(
                theta=param_list[0],
                t=t,
                delay=delay,
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
            ranges=(slice(0.05, 4.0, 0.05), slice(1.05, 10.0, 0.05)),
            full_output=True)
    except (FloatingPointError, OverflowError):
        return inf

    if print_x:
        print("grid search optimal parameter: theta={0}, a={1}".format(
            grid_res[0].tolist()[0], grid_res[0].tolist()[1]))

    return grid_res[1]


def csv_single_param_exp_lower(start_time: int,
                               perform_param: PerformParameter,
                               mc_dist: MonteCarloDist) -> dict:
    total_iterations = 10**3
    valid_iterations = total_iterations
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
        raise NameError("Distribution parameter {0} is infeasible".format(
            mc_dist.mc_enum))

    res_array = np.empty([total_iterations, 3])

    for i in tqdm(range(total_iterations)):
        setting = SingleServerPerform(
            arr=DM1(lamb=param_array[i, 0]),
            const_rate=ConstantRate(rate=param_array[i, 1]),
            perform_param=perform_param)

        theta_bounds = [(0.1, 4.0)]
        bound_array = theta_bounds[:]

        res_array[i, 0] = Optimize(setting=setting).grid_search(
            bound_list=bound_array, delta=delta)

        bound_array_power = theta_bounds[:]
        bound_array_power.append((0.9, 4.0))

        res_array[i, 1] = OptimizeNew(setting_new=setting).grid_search(
            bound_list=bound_array_power, delta=delta)

        if perform_param.perform_metric == PerformEnum.DELAY_PROB:
            res_array[i, 2] = delay_prob_lower_exp_dm1_opt(
                t=start_time,
                delay=perform_param.value,
                lamb=param_array[i, 0],
                rate=param_array[i, 1])

            if res_array[i, 0] >= 1.0:
                res_array[i, ] = nan

        elif perform_param.perform_metric == PerformEnum.OUTPUT:
            res_array[i, 2] = output_lower_exp_dm1_opt(
                s=start_time,
                delta_time=perform_param.value,
                lamb=param_array[i, 0],
                rate=param_array[i, 1])

        else:
            raise NameError("{0} is an infeasible performance metric".format(
                perform_param.perform_metric))

        if (res_array[i, 1] == inf or res_array[i, 2] == inf
                or res_array[i, 0] == nan or res_array[i, 1] == nan
                or res_array[i, 2] == nan):
            res_array[i, ] = nan
            valid_iterations -= 1

    # print("exponential results", res_array[:, 2])

    res_dict = three_col_array_to_results(
        arrival_enum=ArrivalEnum.DM1,
        res_array=res_array,
        valid_iterations=valid_iterations,
        metric=metric)

    res_dict.update({
        "iterations": total_iterations,
        "delta_time": perform_param.value,
        "optimization": "grid_search",
        "metric": "relative",
        "MCDistribution": mc_dist.to_name(),
        "MCParam": mc_dist.param_to_string()
    })

    with open(
            "lower_single_{0}_DM1_results_MC{1}_power_exp.csv".format(
                perform_param.to_name(), mc_dist.to_name()), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in res_dict.items():
            writer.writerow([key, value])

    return res_dict


if __name__ == '__main__':
    START = 30

    # DELTA_TIME = 10
    # DELAY10 = PerformParameter(
    #     perform_metric=PerformEnum.DELAY_PROB, value=DELTA_TIME)

    DELTA_TIME = 4
    OUTPUT4 = PerformParameter(
        perform_metric=PerformEnum.OUTPUT, value=DELTA_TIME)

    # LAMB = 1.0
    # SERVICE_RATE = 1.2
    #
    # BOUND_LIST = [(0.05, 10.0)]
    # BOUND_LIST_NEW = [(0.05, 10.0), (1.05, 20.0)]
    # DELTA = 0.05
    # PRINT_X = False
    #
    # CR_SERVER = ConstantRate(SERVICE_RATE)
    #
    # EXP_ARRIVAL = DM1(lamb=LAMB)
    #
    # DM1_SINGLE = SingleServerPerform(
    #     arr=EXP_ARRIVAL, const_rate=CR_SERVER, perform_param=OUTPUT4)
    #
    # DM1_STANDARD_OPT = Optimize(
    #     setting=DM1_SINGLE, print_x=PRINT_X).grid_search(
    #         bound_list=BOUND_LIST, delta=DELTA)
    # print("DM1 Standard Opt: ", DM1_STANDARD_OPT)
    #
    # DM1_POWER_OPT = OptimizeNew(
    #     setting_new=DM1_SINGLE, print_x=PRINT_X).grid_search(
    #         bound_list=BOUND_LIST_NEW, delta=DELTA)
    # print("DM1 Power Opt: ", DM1_POWER_OPT)
    #
    # DM1_EXP_LOWER_OPT = delay_prob_lower_exp_dm1_opt(
    #     t=START,
    #     delay=DELTA_TIME,
    #     lamb=LAMB,
    #     rate=SERVICE_RATE,
    #     print_x=PRINT_X)
    # print("DM1 Exp Lower Opt: ", DM1_EXP_LOWER_OPT)

    # MC_UNIF20 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[20.0])
    MC_UNIF10 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[10.0])
    MC_EXP1 = MonteCarloDist(mc_enum=MCEnum.EXPONENTIAL, param_list=[1.0])

    def fun1():
        print(
            csv_single_param_exp_lower(
                start_time=START, perform_param=OUTPUT4, mc_dist=MC_UNIF10))

    def fun2():
        print(
            csv_single_param_exp_lower(
                start_time=START, perform_param=OUTPUT4, mc_dist=MC_EXP1))

    def run_in_parallel(*funcs):
        proc = []
        for func in funcs:
            process_instance = Process(target=func)
            process_instance.start()
            proc.append(process_instance)
        for process_instance in proc:
            process_instance.join()

    run_in_parallel(fun1, fun2)
