"""Compute optimal and average improvement for different parameters"""

import csv
from math import inf, nan
from multiprocessing import Process

import numpy as np
from tqdm import tqdm

from library.array_to_results import two_col_array_to_results
from library.compare_old_new import compute_improvement
from library.mc_enum import MCEnum
from library.mc_enum_to_dist import mc_enum_to_dist
from library.monte_carlo_dist import MonteCarloDist
from library.perform_parameter import PerformParameter
from nc_operations.perform_enum import PerformEnum
from nc_processes.arrival_distribution import DM1, EBB, MD1, MMOO
from nc_processes.arrival_enum import ArrivalEnum
from nc_processes.constant_rate_server import ConstantRate
from nc_processes.regulated_arrivals import LeakyBucketMassOne
from optimization.opt_method import OptMethod
from single_server.single_server_perform import SingleServerPerform


def csv_single_param_power(
        arrival_enum: ArrivalEnum, perform_param: PerformParameter,
        opt_method: OptMethod, mc_dist: MonteCarloDist) -> dict:
    """Chooses parameters by Monte Carlo type random choice"""
    total_iterations = 10**4
    valid_iterations = total_iterations
    metric = "relative"

    size_array = [
        total_iterations,
        arrival_enum.number_parameters() + 1  # const_rate has 1 parameter
    ]
    # [rows, columns]

    param_array = mc_enum_to_dist(mc_dist=mc_dist, size=size_array)

    res_array = np.empty([total_iterations, 2])

    for i in tqdm(range(total_iterations), total=total_iterations):
        if arrival_enum == ArrivalEnum.DM1:
            arrival = DM1(lamb=param_array[i, 0])

        elif arrival_enum == ArrivalEnum.MD1:
            arrival = MD1(
                lamb=param_array[i, 0], packet_size=param_array[i, 1])
        #     choose packet size = server rate
        # TODO: check whether this is reasonable

        elif arrival_enum == ArrivalEnum.MMOO:
            arrival = MMOO(
                mu=param_array[i, 0],
                lamb=param_array[i, 1],
                burst=param_array[i, 2])

        elif arrival_enum == ArrivalEnum.EBB:
            arrival = EBB(
                prefactor=param_array[i, 0],
                decay=param_array[i, 1],
                rho_single=param_array[i, 2])

        elif arrival_enum == ArrivalEnum.MassOne:
            arrival = LeakyBucketMassOne(
                sigma_single=param_array[i, 0],
                rho_single=param_array[i, 1],
                n=20)
        # TODO: note that n is fixed

        else:
            raise NameError("Arrival parameter {0} is infeasible".format(
                arrival_enum.name))

        setting = SingleServerPerform(
            arr=arrival,
            const_rate=ConstantRate(
                rate=param_array[i, arrival_enum.number_parameters()]),
            perform_param=perform_param)

        # standard_bound, new_bound = compute_improvement()
        res_array[i, 0], res_array[i, 1] = compute_improvement(
            setting=setting, opt_method=opt_method)

        if perform_param.perform_metric == PerformEnum.DELAY_PROB:
            if res_array[i, 1] >= 1.0:
                res_array[i, ] = nan

        if (res_array[i, 0] == inf or res_array[i, 1] == inf
                or res_array[i, 0] == nan or res_array[i, 1] == nan):
            res_array[i, ] = nan
            valid_iterations -= 1

    res_dict = two_col_array_to_results(
        arrival_enum=arrival_enum,
        metric=metric,
        param_array=param_array,
        res_array=res_array,
        number_servers=1,
        valid_iterations=valid_iterations)

    res_dict.update({
        "iterations": total_iterations,
        "delta_time": perform_param.value,
        "optimization": opt_method.name,
        "metric": metric,
        "MCDistribution": mc_dist.to_name(),
        "MCParam": mc_dist.param_to_string()
    })

    with open(
            "single_{0}_{1}_results_MC{2}_{3}_{4}.csv".format(
                perform_param.to_name(), arrival_enum.name, mc_dist.to_name(),
                opt_method.name, metric), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in res_dict.items():
            writer.writerow([key, value])

    return res_dict


def grid_param_single_dm1(perform_param: PerformParameter,
                          opt_method: OptMethod, metric: str, lamb1_range,
                          rate1_range) -> dict:
    """Choose parameters along a grid"""

    total_iterations = len(lamb1_range) * len(rate1_range)
    valid_iterations = total_iterations

    param_array = np.empty([total_iterations, 2])
    res_array = np.empty([total_iterations, 2])

    i = 0
    for lamb1 in tqdm(lamb1_range):
        for rate1 in rate1_range:
            setting = SingleServerPerform(
                arr=DM1(lamb=lamb1),
                const_rate=ConstantRate(rate=rate1),
                perform_param=perform_param)
            param_array[i, 0] = lamb1
            param_array[i, 1] = rate1

            res_array[i, 0], res_array[i, 1] = compute_improvement(
                setting=setting, opt_method=opt_method, number_l=1)
            if res_array[i, 1] == inf:
                res_array[i, ] = nan
                valid_iterations -= 1

            i += 1

    return two_col_array_to_results(
        arrival_enum=ArrivalEnum.DM1,
        metric=metric,
        param_array=param_array,
        res_array=param_array,
        number_servers=1,
        valid_iterations=valid_iterations)


if __name__ == '__main__':
    # OUTPUT4 = PerformParameter(perform_metric=PerformEnum.OUTPUT, value=4)

    DELAY10 = PerformParameter(perform_metric=PerformEnum.DELAY_PROB, value=10)

    COMMON_OPTIMIZATION = OptMethod.GRID_SEARCH

    MC_UNIF20 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[20.0])
    MC_EXP1 = MonteCarloDist(mc_enum=MCEnum.EXPONENTIAL, param_list=[1.0])

    ARRIVAL_PROCESS = ArrivalEnum.MD1

    def fun1():
        print(
            csv_single_param_power(
                arrival_enum=ARRIVAL_PROCESS,
                perform_param=DELAY10,
                opt_method=COMMON_OPTIMIZATION,
                mc_dist=MC_UNIF20))

    def fun2():
        print(
            csv_single_param_power(
                arrival_enum=ARRIVAL_PROCESS,
                perform_param=DELAY10,
                opt_method=COMMON_OPTIMIZATION,
                mc_dist=MC_EXP1))

    def run_in_parallel(*funcs):
        proc = []
        for func in funcs:
            process_instance = Process(target=func)
            process_instance.start()
            proc.append(process_instance)
        for process_instance in proc:
            process_instance.join()

    run_in_parallel(fun1, fun2)

    # print(
    #     grid_param_single_dm1(
    #         perform_param=DELAY10,
    #         opt_method=OptMethod.GRID_SEARCH,
    #         lamb1_range=[0.1, 0.3, 0.5, 0.7, 1, 2, 4, 8, 12],
    #         rate1_range=[0.1, 0.3, 0.5, 0.7, 1, 2, 4, 8, 12]))
