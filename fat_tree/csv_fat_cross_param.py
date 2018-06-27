"""Compute optimal and average improvement for different parameters."""

import csv
from math import floor, nan
from multiprocessing import Process

import numpy as np

from fat_tree.fat_cross_perform import FatCrossPerform
from library.array_to_results import data_array_to_results
from library.compare_old_new import compute_improvement
from library.mc_name import MCName
from library.monte_carlo_dist import MonteCarloDist
from library.perform_parameter import PerformParameter
from nc_operations.perform_metric import PerformMetric
from nc_processes.arrival_distribution import (MMOO, ArrivalDistribution,
                                               ExponentialArrival)
from nc_processes.constant_rate_server import ConstantRate
from optimization.opt_method import OptMethod

########################################################################
# Find Optimal Parameters
########################################################################


def csv_fat_cross_param(arrival: ArrivalDistribution,
                        const_rate: ConstantRate,
                        number_servers: int,
                        perform_param: PerformParameter,
                        opt_method: OptMethod,
                        mc_dist: MonteCarloDist) -> dict:
    """Chooses parameters by Monte Carlo type random choice."""
    total_iterations = 10**4
    metric = "relative"

    size_array = [
        total_iterations,
        (arrival.number_parameters() + const_rate.number_parameters()) *
        number_servers
    ]
    # [rows, columns]

    if mc_dist.mc_dist_name == MCName.UNIFORM:
        param_array = np.random.uniform(
            low=0, high=mc_dist.param_list[0], size=size_array)
    elif mc_dist.mc_dist_name == MCName.EXPONENTIAL:
        param_array = np.random.exponential(
            scale=mc_dist.param_list[0], size=size_array)
    else:
        raise ValueError("Distribution parameter {0} is infeasible".format(
            mc_dist.mc_dist_name))

    res_array = np.empty([total_iterations, 2])

    # print(res_array)

    for i in range(total_iterations):
        if isinstance(arrival, ExponentialArrival):
            arrive_list = [
                ExponentialArrival(lamb=param_array[i, j])
                for j in range(number_servers)
            ]

        elif isinstance(arrival, MMOO):
            arrive_list = [
                MMOO(
                    mu=param_array[i, j],
                    lamb=param_array[i, number_servers + j],
                    burst=param_array[i, 2 * number_servers + j])
                for j in range(number_servers)
            ]

        else:
            raise NameError("Arrival parameter {0} is infeasible".format(
                arrival.__class__.__name__))

        service_list = [
            ConstantRate(rate=param_array[
                i, arrival.number_parameters() * number_servers + j])
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

        if res_array[i, 1] >= 1:
            res_array[i, ] = nan

        if i % floor(total_iterations / 10) == 0:
            print("iteration {0} of {1}".format(i, total_iterations))

    res_dict = data_array_to_results(
        arrival=arrival,
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
        "MCDistribution": mc_dist.mc_dist_name.name,
        "MCParam": mc_dist.param_to_string(),
        "number_servers": number_servers
    })

    with open(
            "sim_" + perform_param.perform_metric.name + "_" +
            arrival.__class__.__name__ + "_results_MC" +
            mc_dist.mc_dist_name.name + "_" + opt_method.name + "_" + metric +
            ".csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in res_dict.items():
            writer.writerow([key, value])

    return res_dict


if __name__ == '__main__':
    DELAY_PROB10 = PerformParameter(
        perform_metric=PerformMetric.DELAY_PROB, value=10)

    EXP_ARRIVAL1 = ExponentialArrival()
    MMOO_ARRIVAL1 = MMOO()
    CONST_RATE1 = ConstantRate()

    COMMON_OPTIMIZATION = OptMethod.GRID_SEARCH

    MC_UNIF20 = MonteCarloDist(mc_dist_name=MCName.UNIFORM, param_list=[20.0])
    MC_EXP1 = MonteCarloDist(MCName.EXPONENTIAL, [1.0])

    def fun1():
        print(
            csv_fat_cross_param(
                arrival=MMOO_ARRIVAL1,
                const_rate=CONST_RATE1,
                number_servers=2,
                perform_param=DELAY_PROB10,
                opt_method=COMMON_OPTIMIZATION,
                mc_dist=MC_EXP1))

    def fun2():
        print(
            csv_fat_cross_param(
                arrival=EXP_ARRIVAL1,
                const_rate=CONST_RATE1,
                number_servers=2,
                perform_param=DELAY_PROB10,
                opt_method=COMMON_OPTIMIZATION,
                mc_dist=MC_EXP1))


    def fun3():
        print(
            csv_fat_cross_param(
                arrival=MMOO_ARRIVAL1,
                const_rate=CONST_RATE1,
                number_servers=2,
                perform_param=DELAY_PROB10,
                opt_method=COMMON_OPTIMIZATION,
                mc_dist=MC_UNIF20))

    def fun4():
        print(
            csv_fat_cross_param(
                arrival=EXP_ARRIVAL1,
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
