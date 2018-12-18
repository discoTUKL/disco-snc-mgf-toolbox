"""Compare with alternative traffic description"""

import csv
from math import exp, inf, log, nan
from multiprocessing import Process
from typing import List

import numpy as np
import scipy.optimize
from tqdm import tqdm

from utils.array_to_results import three_col_array_to_results
from utils.exceptions import ParameterOutOfBounds
from utils.mc_enum import MCEnum
from utils.monte_carlo_dist import MonteCarloDist
from utils.perform_parameter import PerformParameter
from nc_operations.perform_enum import PerformEnum
from nc_arrivals.arrival_enum import ArrivalEnum
from nc_arrivals.arrivals_alternative import expect_dm1
from nc_service.constant_rate_server import ConstantRate
from nc_arrivals.qt import DM1
from nc_service.service_alternative import expect_const_rate
from optimization.optimize import Optimize
from optimization.optimize_new import OptimizeNew
from single_server.single_server_perform import SingleServerPerform


def f_exp(theta: float, i: int, s: int, t: int, lamb: float, rate: float,
          a: float) -> float:
    return a**(exp(theta * (expect_dm1(delta_time=t - i, lamb=lamb) -
                            expect_const_rate(delta_time=s - i, rate=rate))))


def f_sample(i: int, s: int, t: int, lamb: float, rate: float) -> float:
    res: float = np.sum(
        np.random.exponential(scale=1 / lamb, size=t - i) - rate * (s - i))
    return res


def output_lower_exp_dm1(theta: float, s: int, delta_time: int, lamb: float,
                         rate: float, a: float) -> float:
    if 1 / lamb >= rate:
        raise ParameterOutOfBounds(
            (f"The arrivals' long term rate {1 / lamb} has to be smaller than"
             f"the service's long term rate {rate}"))

    if theta <= 0:
        raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

    if a <= 1:
        raise ParameterOutOfBounds(f"base a = {a} must be >0")

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
                             print_x=False) -> float:
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
                                 print_x=False) -> float:
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


def delay_prob_sample_exp_dm1(theta: float, t: int, delay: int, lamb: float,
                              rate: float, a: float,
                              sample_size: int) -> float:
    if 1 / lamb >= rate:
        raise ParameterOutOfBounds(
            ("The arrivals' long term rate {0} has to be smaller than"
             "the service's long term rate {1}").format(1 / lamb, rate))

    if theta <= 0:
        raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

    if a <= 1:
        raise ParameterOutOfBounds("base a={0} must be >0".format(a))

    res = 0.0

    for j in range(sample_size):
        sum_i = 0.0
        for i in range(t + 1):
            sum_i += a**(exp(
                theta * f_sample(i=i, s=t + delay, t=t, lamb=lamb, rate=rate)))
        res += sum_i

    res /= sample_size

    return res


def delay_prob_sample_exp_dm1_opt(t: int,
                                  delay: int,
                                  lamb: float,
                                  rate: float,
                                  sample_size: int,
                                  print_x=False) -> float:
    def helper_fun(param_list: List[float]) -> float:
        try:
            return delay_prob_sample_exp_dm1(
                theta=param_list[0],
                t=t,
                delay=delay,
                lamb=lamb,
                rate=rate,
                a=param_list[1],
                sample_size=sample_size)
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
                               mc_dist: MonteCarloDist,
                               sample=False) -> dict:
    total_iterations = 10**2
    valid_iterations = total_iterations
    metric = "relative"
    sample_size = 10**3

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
    if sample:
        res_array_sample = np.empty([total_iterations, 3])

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

            if sample:
                res_array_sample[i, 0] = res_array[i, 0]
                res_array_sample[i, 1] = res_array[i, 1]
                res_array_sample[i, 2] = delay_prob_sample_exp_dm1_opt(
                    t=start_time,
                    delay=perform_param.value,
                    lamb=param_array[i, 0],
                    rate=param_array[i, 1],
                    sample_size=sample_size)

            if res_array[i, 0] > 1.0:
                res_array[i, ] = nan
                if sample:
                    res_array_sample[i, ] = nan

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
            res_array_sample[i, ] = nan
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

    res_dict_sample = three_col_array_to_results(
        arrival_enum=ArrivalEnum.DM1,
        res_array=res_array_sample,
        valid_iterations=valid_iterations,
        metric=metric)

    res_dict_sample.update({
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
        writer = csv.writer(fileobj=csv_file)
        for key, value in res_dict.items():
            writer.writerow([key, value])
    if sample:
        with open(
                "sample_single_{0}_DM1_results_MC{1}_power_exp.csv".format(
                    perform_param.to_name(), mc_dist.to_name()),
                'w') as csv_file:
            writer = csv.writer(fileobj=csv_file)
            for key, value in res_dict_sample.items():
                writer.writerow([key, value])

    return res_dict


# def helper_parallel(index: int) -> (np.ndarray, int):
#     size_array = [TOTAL_ITERATIONS, 2]
#     # [rows, columns]
#     mc_dist = MC_UNIF10
#     perform_param = DELAY10
#
#     if mc_dist.mc_enum == MCEnum.UNIFORM:
#         param_array = np.random.uniform(
#             low=0, high=mc_dist.param_list[0], size=size_array)
#     elif mc_dist.mc_enum == MCEnum.EXPONENTIAL:
#         param_array = np.random.exponential(
#             scale=mc_dist.param_list[0], size=size_array)
#     else:
#         raise NameError("Distribution parameter {0} is infeasible".format(
#             mc_dist.mc_enum))
#
#     res_array = np.empty([TOTAL_ITERATIONS, 3])
#     valid_iterations = TOTAL_ITERATIONS
#
#     setting = SingleServerPerform(
#         arr=DM1(lamb=param_array[index, 0]),
#         const_rate=ConstantRate(rate=param_array[index, 1]),
#         perform_param=perform_param)
#
#     theta_bounds = [(0.1, 4.0)]
#     bound_array = theta_bounds[:]
#
#     res_array[index, 0] = Optimize(setting=setting).grid_search(
#         bound_list=bound_array, delta=DELTA)
#
#     bound_array_power = theta_bounds[:]
#     bound_array_power.append((0.9, 4.0))
#
#     res_array[index, 1] = OptimizeNew(setting_new=setting).grid_search(
#         bound_list=bound_array_power, delta=DELTA)
#
#     if perform_param.perform_metric == PerformEnum.DELAY_PROB:
#         res_array[index, 2] = delay_prob_lower_exp_dm1_opt(
#             t=START,
#             delay=perform_param.value,
#             lamb=param_array[index, 0],
#             rate=param_array[index, 1])
#
#         if res_array[index, 0] > 1.0:
#             res_array[index, ] = nan
#
#     elif perform_param.perform_metric == PerformEnum.OUTPUT:
#         res_array[index, 2] = output_lower_exp_dm1_opt(
#             s=START,
#             delta_time=perform_param.value,
#             lamb=param_array[index, 0],
#             rate=param_array[index, 1])
#
#     else:
#         raise NameError("{0} is an infeasible performance metric".format(
#             perform_param.perform_metric))
#
#     if (res_array[index, 1] == inf or res_array[index, 2] == inf
#             or res_array[index, 0] == nan or res_array[index, 1] == nan
#             or res_array[index, 2] == nan):
#         res_array[index, ] = nan
#         valid_iterations -= 1
#
#     return res_array, valid_iterations
#
#
# def csv_single_param_exp_lower_par(perform_param: PerformParameter,
#                                    mc_dist: MonteCarloDist) -> dict:
#     with Pool() as p:
#         resulting_array, valid_iter = list(
#             tqdm(
#                 p.imap(func=helper_parallel, iterable=range(TOTAL_ITERATIONS)),
#                 total=TOTAL_ITERATIONS))
#
#     res_dict = three_col_array_to_results(
#         arrival_enum=ArrivalEnum.DM1,
#         res_array=resulting_array,
#         valid_iterations=valid_iter,
#         metric=METRIC)
#
#     res_dict.update({
#         "iterations": TOTAL_ITERATIONS,
#         "delta_time": perform_param.value,
#         "optimization": "grid_search",
#         "metric": "relative",
#         "MCDistribution": mc_dist.to_name(),
#         "MCParam": mc_dist.param_to_string()
#     })
#
#     with open(
#             "lower_single_{0}_DM1_results_MC{1}_power_exp.csv".format(
#                 perform_param.to_name(), mc_dist.to_name()), 'w') as csv_file:
#         writer = csv.writer(fileobj=csv_file)
#         for key, value in res_dict.items():
#             writer.writerow([key, value])
#
#     return res_dict

if __name__ == '__main__':
    # TOTAL_ITERATIONS = 10**3
    TOTAL_ITERATIONS = 200
    METRIC = "relative"

    DELTA = 0.05

    START = 30

    DELTA_TIME = 10
    DELAY10 = PerformParameter(
        perform_metric=PerformEnum.DELAY_PROB, value=DELTA_TIME)

    # DELTA_TIME = 4
    # OUTPUT4 = PerformParameter(
    #     perform_metric=PerformEnum.OUTPUT, value=DELTA_TIME)

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
                start_time=START,
                perform_param=DELAY10,
                mc_dist=MC_UNIF10,
                sample=False))

    def fun2():
        print(
            csv_single_param_exp_lower(
                start_time=START,
                perform_param=DELAY10,
                mc_dist=MC_EXP1,
                sample=False))

    def run_in_parallel(*funcs):
        proc = []
        for func in funcs:
            process_instance = Process(target=func)
            process_instance.start()
            proc.append(process_instance)
        for process_instance in proc:
            process_instance.join()

    run_in_parallel(fun1, fun2)
