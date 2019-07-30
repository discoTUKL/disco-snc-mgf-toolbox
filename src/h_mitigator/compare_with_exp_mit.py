"""Compare with alternative traffic description"""

import csv
from math import inf
from typing import List

import numpy as np
import scipy.optimize
from tqdm import tqdm

from bound_evaluation.change_enum import ChangeEnum
from bound_evaluation.mc_enum import MCEnum
from bound_evaluation.monte_carlo_dist import MonteCarloDist
from h_mitigator.array_to_results import two_col_array_to_results
from h_mitigator.single_server_mit_perform import SingleServerMitPerform
from nc_arrivals.arrival_enum import ArrivalEnum
from nc_arrivals.arrivals_alternative import expect_dm1
from nc_arrivals.qt import DM1
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from nc_server.server_alternative import expect_const_rate
from optimization.optimize import Optimize
from utils.exceptions import ParameterOutOfBounds
from utils.perform_parameter import PerformParameter


def f_exp(theta: float, i: int, s: int, t: int, lamb: float, rate: float,
          a: float) -> float:
    return a**(np.exp(theta *
                      (expect_dm1(delta_time=t - i, lamb=lamb) -
                       expect_const_rate(delta_time=s - i, rate=rate))))


def f_sample(i: int, s: int, t: int, lamb: float, rate: float) -> np.ndarray:
    return np.sum(
        np.random.exponential(scale=1 / lamb, size=t - i) - rate * (s - i))


def delay_prob_lower_exp_dm1(theta: float, t: int, delay: int, lamb: float,
                             rate: float, a: float) -> float:
    if 1 / lamb >= rate:
        raise ParameterOutOfBounds(
            (f"The arrivals' long term rate {1 / lamb} has to be smaller than"
             f"the service's long term rate {rate}"))

    if theta <= 0:
        raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

    if a <= 1:
        raise ParameterOutOfBounds(f"base a = {a} must be >1")

    sum_j = 0.0

    for _i in range(t + 1):
        try:
            summand = f_exp(theta=theta,
                            i=_i,
                            s=t + delay,
                            t=t,
                            lamb=lamb,
                            rate=rate,
                            a=a)
        except (FloatingPointError, OverflowError):
            summand = inf

        sum_j += summand

    return np.log(sum_j) / np.log(a)


def delay_prob_lower_exp_dm1_opt(t: int,
                                 delay: int,
                                 lamb: float,
                                 rate: float,
                                 print_x=False) -> float:
    def helper_fun(param_list: List[float]) -> float:
        try:
            return delay_prob_lower_exp_dm1(theta=param_list[0],
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
        grid_res = scipy.optimize.brute(func=helper_fun,
                                        ranges=(slice(0.05, 4.0, 0.05),
                                                slice(1.05, 10.0, 0.05)),
                                        full_output=True)
    except (FloatingPointError, OverflowError):
        return inf

    if print_x:
        print(
            f"grid search optimal parameter: theta={grid_res[0].tolist()[0]},"
            f" a={grid_res[0].tolist()[1]}")

    return grid_res[1]


def delay_prob_sample_exp_dm1(theta: float, t: int, delay: int, lamb: float,
                              rate: float, a: float,
                              sample_size: int) -> float:
    if 1 / lamb >= rate:
        raise ParameterOutOfBounds(
            (f"The arrivals' long term rate={1 / lamb} has to be smaller than "
             f"the service's long term rate={rate}"))

    if theta <= 0:
        raise ParameterOutOfBounds(f"theta={theta} must be > 0")

    if a <= 1:
        raise ParameterOutOfBounds(f"base a={a} must be >0")

    res = 0.0

    for j in range(sample_size):
        sum_i = 0.0
        for i in range(t + 1):
            sum_i += a**(np.exp(
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
            return delay_prob_sample_exp_dm1(theta=param_list[0],
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
        grid_res = scipy.optimize.brute(func=helper_fun,
                                        ranges=(slice(0.05, 4.0, 0.05),
                                                slice(1.05, 10.0, 0.05)),
                                        full_output=True)
    except (FloatingPointError, OverflowError):
        return inf

    if print_x:
        print("grid search optimal parameter: "
              f"theta={grid_res[0].tolist()[0]}, a={grid_res[0].tolist()[1]}")

    return grid_res[1]


def csv_single_param_exp(start_time: int,
                         delay: int,
                         mc_dist: MonteCarloDist,
                         target_util: float,
                         total_iterations: int,
                         sample=False) -> dict:
    valid_iterations = total_iterations
    metric = ChangeEnum.RATIO_REF_NEW
    sample_size = 10**2

    delta = 0.05

    size_array = [total_iterations, 2]
    # [rows, columns]

    if mc_dist.mc_enum == MCEnum.UNIFORM:
        param_array = np.random.uniform(low=0,
                                        high=mc_dist.param_list[0],
                                        size=size_array)
    elif mc_dist.mc_enum == MCEnum.EXPONENTIAL:
        param_array = np.random.exponential(scale=mc_dist.param_list[0],
                                            size=size_array)
    else:
        raise NameError(
            f"Distribution parameter {mc_dist.mc_enum} is infeasible")

    res_array = np.empty([total_iterations, 2])
    res_array_sample = np.empty([total_iterations, 2])

    for i in tqdm(range(total_iterations)):
        single_setting = SingleServerMitPerform(
            arr=DM1(lamb=param_array[i, 0]),
            const_rate=ConstantRateServer(rate=param_array[i, 1]),
            perform_param=PerformParameter(
                perform_metric=PerformEnum.DELAY_PROB, value=delay))

        computation_necessary = True

        # print(res_array[i, ])
        if target_util > 0.0:
            util = single_setting.approximate_utilization()
            if util < target_util or util > 1:
                res_array[i, ] = np.nan
                computation_necessary = False

        if computation_necessary:

            theta_bounds = [(0.1, 4.0)]

            res_array[i, 0] = Optimize(setting=single_setting).grid_search(
                bound_list=theta_bounds, delta=delta)

            res_array[i, 1] = delay_prob_lower_exp_dm1_opt(
                t=start_time,
                delay=delay,
                lamb=param_array[i, 0],
                rate=param_array[i, 1])

            if sample:
                res_array_sample[i, 1] = delay_prob_sample_exp_dm1_opt(
                    t=start_time,
                    delay=delay,
                    lamb=param_array[i, 0],
                    rate=param_array[i, 1],
                    sample_size=sample_size)

            if res_array[i, 0] > 1.0:
                res_array[i, ] = np.nan
                if sample:
                    res_array_sample[i, ] = np.nan

        if np.isnan(res_array[i, 0]) or np.isnan(res_array[i, 1]):
            res_array[i, ] = np.nan
            res_array_sample[i, ] = np.nan
            valid_iterations -= 1

    # print("exponential results", res_array[:, 2])

    res_dict = two_col_array_to_results(arrival_enum=ArrivalEnum.DM1,
                                        param_array=param_array,
                                        res_array=res_array,
                                        number_servers=1,
                                        valid_iterations=valid_iterations,
                                        compare_metric=metric)

    res_dict.update({
        "iterations": total_iterations,
        "delta_time": delay,
        "optimization": "grid_search",
        "metric": metric.name,
        "MCDistribution": mc_dist.to_name(),
        "MCParam": mc_dist.param_to_string()
    })

    if sample:
        res_array_sample[:, 0] = res_array[:, 0]

        res_dict_sample = two_col_array_to_results(
            arrival_enum=ArrivalEnum.DM1,
            param_array=param_array,
            res_array=res_array_sample,
            number_servers=1,
            valid_iterations=valid_iterations,
            compare_metric=metric)

        res_dict_sample.update({
            "iterations": total_iterations,
            "delta_time": delay,
            "optimization": "grid_search",
            "metric": metric.name,
            "MCDistribution": mc_dist.to_name(),
            "MCParam": mc_dist.param_to_string()
        })

    suffix = f"single_DELAY_PROB_DM1_results" \
        f"_MC{mc_dist.to_name()}_power_exp.csv"

    with open("lower_" + suffix, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in res_dict.items():
            writer.writerow([key, value])
    if sample:
        with open("sample_" + suffix, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in res_dict_sample.items():
                writer.writerow([key, value])

    return res_dict


# def helper_parallel(index: int) -> (np.ndarray, int):
#     size_array = [TOTAL_ITERATIONS, 2]
#     # [rows, columns]
#     mc_dist = MC_UNIF10
#
#     if mc_dist.mc_enum == MCEnum.UNIFORM:
#         param_array = np.random.uniform(low=0,
#                                         high=mc_dist.param_list[0],
#                                         size=size_array)
#     elif mc_dist.mc_enum == MCEnum.EXPONENTIAL:
#         param_array = np.random.exponential(scale=mc_dist.param_list[0],
#                                             size=size_array)
#     else:
#         raise NameError(
#             f"Distribution parameter {mc_dist.mc_enum} is infeasible")
#
#     res_array = np.empty([TOTAL_ITERATIONS, 3])
#     valid_iterations = TOTAL_ITERATIONS
#
#     setting = SingleServerMitPerform(
#         arr=DM1(lamb=param_array[index, 0]),
#         const_rate=ConstantRateServer(rate=param_array[index, 1]),
#         perform_param=PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
#                                        value=DELAY))
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
#     res_array[index, 1] = OptimizeMitigator(setting_h_mit=setting).grid_search(
#         bound_list=bound_array_power, delta=DELTA)
#
#     res_array[index, 2] = delay_prob_lower_exp_dm1_opt(
#         t=START,
#         delay=DELAY,
#         lamb=param_array[index, 0],
#         rate=param_array[index, 1])
#
#     if res_array[index, 0] > 1.0:
#         res_array[index, ] = np.nan
#
#     if (res_array[index, 1] == inf or res_array[index, 2] == inf
#             or np.isnan(res_array[index, 0]) or np.isnan(res_array[index, 1])
#             or np.isnan(res_array[index, 2])):
#         res_array[index, ] = np.nan
#         valid_iterations -= 1
#
#     return res_array, valid_iterations
#
#
# def csv_single_param_exp_lower_par(perform_param: PerformParameter,
#                                    mc_dist: MonteCarloDist) -> dict:
#     with Pool() as p:
#         resulting_array, valid_iter = list(
#             tqdm(p.imap(func=helper_parallel,
#                         iterable=range(TOTAL_ITERATIONS)),
#                  total=TOTAL_ITERATIONS))
#
#     res_dict = three_col_array_to_results(arrival_enum=ArrivalEnum.DM1,
#                                           res_array=resulting_array,
#                                           valid_iterations=valid_iter,
#                                           compare_metric=CHANGE_METRIC)
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
    CHANGE_METRIC = ChangeEnum.RATIO_NEW_REF

    DELTA = 0.05

    START = 30

    DELAY = 10

    # LAMB = 1.0
    # SERVICE_RATE = 1.2
    #
    # BOUND_LIST = [(0.05, 10.0)]
    # BOUND_LIST_NEW = [(0.05, 10.0), (1.05, 20.0)]
    # DELTA = 0.05
    # PRINT_X = False
    #
    # CR_SERVER = ConstantRateServer(SERVICE_RATE)
    #
    # EXP_ARRIVAL = DM1(lamb=LAMB)
    #
    # DM1_SINGLE = SingleServerMitPerform(
    #     arr=EXP_ARRIVAL,
    #     const_rate=CR_SERVER,
    #     perform_param=PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
    #                                    value=DELTA_TIME))
    #
    # DM1_STANDARD_OPT = Optimize(
    #     setting=DM1_SINGLE, print_x=PRINT_X).grid_search(bound_list=BOUND_LIST,
    #                                                      delta=DELTA)
    # print("DM1 Standard Opt: ", DM1_STANDARD_OPT)
    #
    # DM1_POWER_OPT = OptimizeMitigator(setting_h_mit=DM1_SINGLE,
    #                                   print_x=PRINT_X).grid_search(
    #                                       bound_list=BOUND_LIST_NEW,
    #                                       delta=DELTA)
    # print("DM1 Power Opt: ", DM1_POWER_OPT)
    #
    # DM1_EXP_LOWER_OPT = delay_prob_lower_exp_dm1_opt(t=START,
    #                                                  delay=DELTA_TIME,
    #                                                  lamb=LAMB,
    #                                                  rate=SERVICE_RATE,
    #                                                  print_x=PRINT_X)
    # print("DM1 Exp Lower Opt: ", DM1_EXP_LOWER_OPT)

    # MC_UNIF20 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[20.0])
    MC_UNIF10 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[10.0])
    MC_EXP1 = MonteCarloDist(mc_enum=MCEnum.EXPONENTIAL, param_list=[1.0])

    print(
        csv_single_param_exp(start_time=START,
                             delay=DELAY,
                             mc_dist=MC_UNIF10,
                             target_util=0.0,
                             total_iterations=10**4,
                             sample=False))

    print(
        csv_single_param_exp(start_time=START,
                             delay=DELAY,
                             mc_dist=MC_EXP1,
                             target_util=0.0,
                             total_iterations=10**4,
                             sample=False))
