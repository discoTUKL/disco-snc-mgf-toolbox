"""Compute optimal and average improvement for different parameters."""

import csv
from math import floor, nan
from multiprocessing import Process

import numpy as np

from fat_tree.fat_cross_perform import FatCrossPerform
from library.array_to_results import data_array_to_results
from library.compare_old_new import compute_improvement
from library.mc_enum import MCEnum
from library.monte_carlo_dist import MonteCarloDist
from library.perform_parameter import PerformParameter
from nc_operations.perform_enum import PerformEnum
from nc_processes.arrival_distribution import DM1, MMOO
from nc_processes.arrival_enum import ArrivalEnum
from nc_processes.constant_rate_server import ConstantRate
from optimization.opt_method import OptMethod

########################################################################
# Find Optimal Parameters
########################################################################


def csv_fat_cross_param(arrival_enum: ArrivalEnum, const_rate: ConstantRate,
                        number_servers: int, perform_param: PerformParameter,
                        opt_method: OptMethod,
                        mc_dist: MonteCarloDist) -> dict:
    """Chooses parameters by Monte Carlo type random choice."""
    total_iterations = 10**4
    metric = "relative"

    size_array = [
        total_iterations,
        (arrival_enum.number_parameters() + 1) * number_servers
        # const_rate has 1 parameter
    ]
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

    res_array = np.empty([total_iterations, 2])

    # print(res_array)

    for i in range(total_iterations):
        if arrival_enum == ArrivalEnum.DM1:
            arrive_list = [
                DM1(lamb=param_array[i, j]) for j in range(number_servers)
            ]

        elif arrival_enum == ArrivalEnum.MMOO:
            arrive_list = [
                MMOO(
                    mu=param_array[i, j],
                    lamb=param_array[i, number_servers + j],
                    burst=param_array[i, 2 * number_servers + j])
                for j in range(number_servers)
            ]

        else:
            raise NameError("Arrival parameter {0} is infeasible".format(
                arrival_enum.name))

        service_list = [
            ConstantRate(rate=param_array[
                i, arrival_enum.number_parameters() * number_servers + j])
            for j in range(number_servers)
        ]

        setting = FatCrossPerform(
            arr_list=arrive_list,
            ser_list=service_list,
            perform_param=perform_param)

        # print(res_array[i, ])

        # standard_bound, new_bound = compute_improvement()
        res_array[i, 0], res_array[i, 1] = compute_improvement(
            setting=setting,
            opt_method=opt_method,
            number_l=number_servers - 1)

        if (res_array[i, 1] >= 1 or res_array[i, 0] == nan
                or res_array[i, 1] == nan):
            res_array[i, ] = nan

        if i % floor(total_iterations / 10) == 0:
            print("iteration {0} of {1}".format(i, total_iterations))

    res_dict = data_array_to_results(
        arrival_enum=arrival_enum,
        const_rate=const_rate,
        param_array=param_array,
        res_array=res_array,
        number_servers=number_servers,
        metric=metric)

    res_dict.update({
        "T": perform_param.value,
        "optimization": opt_method.name,
        "metric": metric,
        "iterations": total_iterations,
        "MCDistribution": mc_dist.to_name(),
        "MCParam": mc_dist.param_to_string(),
        "number_servers": number_servers
    })

    with open(
            "sim_{0}_{1}_results_MC{2}_{3}_{4}.csv".format(
                perform_param.to_name(), arrival_enum.name, mc_dist.to_name(),
                opt_method.name, metric), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in res_dict.items():
            writer.writerow([key, value])

    return res_dict


if __name__ == '__main__':
    DELAY_PROB10 = PerformParameter(
        perform_metric=PerformEnum.DELAY_PROB, value=10)
    CONST_RATE1 = ConstantRate()

    COMMON_OPTIMIZATION = OptMethod.GRID_SEARCH

    MC_UNIF20 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[20.0])
    MC_EXP1 = MonteCarloDist(mc_enum=MCEnum.EXPONENTIAL, param_list=[1.0])

    def fun1():
        print(
            csv_fat_cross_param(
                arrival_enum=ArrivalEnum.MMOO,
                const_rate=CONST_RATE1,
                number_servers=2,
                perform_param=DELAY_PROB10,
                opt_method=COMMON_OPTIMIZATION,
                mc_dist=MC_EXP1))

    def fun2():
        print(
            csv_fat_cross_param(
                arrival_enum=ArrivalEnum.DM1,
                const_rate=CONST_RATE1,
                number_servers=2,
                perform_param=DELAY_PROB10,
                opt_method=COMMON_OPTIMIZATION,
                mc_dist=MC_EXP1))

    def fun3():
        print(
            csv_fat_cross_param(
                arrival_enum=ArrivalEnum.MMOO,
                const_rate=CONST_RATE1,
                number_servers=2,
                perform_param=DELAY_PROB10,
                opt_method=COMMON_OPTIMIZATION,
                mc_dist=MC_UNIF20))

    def fun4():
        print(
            csv_fat_cross_param(
                arrival_enum=ArrivalEnum.DM1,
                const_rate=CONST_RATE1,
                number_servers=2,
                perform_param=DELAY_PROB10,
                opt_method=COMMON_OPTIMIZATION,
                mc_dist=MC_UNIF20))

    def run_in_parallel(*funcs):
        """Run auxiliary functions in parallel."""
        proc = []
        for func in funcs:
            process_instance = Process(target=func)
            process_instance.start()
            proc.append(process_instance)
        for process_instance in proc:
            process_instance.join()

    run_in_parallel(fun1, fun2, fun3, fun4)
