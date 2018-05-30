"""Compute optimal and average improvement for different parameters"""

import csv
from math import floor, inf, nan
from multiprocessing import Process

import numpy as np

from library.array_to_results import data_array_to_results
from library.compare_old_new import compute_improvement
from library.mc_name import MCName
from library.monte_carlo_dist import MonteCarloDist
from library.perform_parameter import PerformParameter
from nc_operations.perform_metric import PerformMetric
from nc_processes.arrival_distribution import (MMOO, ArrivalDistribution,
                                               ExponentialArrival)
from nc_processes.service_distribution import ConstantRate, ServiceDistribution
from optimization.opt_method import OptMethod
from single_server.single_server_perform import SingleServerPerform


def csv_single_server_param(arrival: ArrivalDistribution,
                            service: ServiceDistribution,
                            perform_param: PerformParameter,
                            opt_method: OptMethod,
                            total_iterations: int,
                            mc_dist: MonteCarloDist,
                            metric="relative") -> dict:
    """Chooses parameters by Monte Carlo type random choice"""

    # 1 Parameter for service
    size_array = [total_iterations, arrival.number_parameters() + 1]
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

    for i in range(total_iterations):
        if isinstance(arrival, ExponentialArrival):
            setting = SingleServerPerform(
                arr=ExponentialArrival(lamb=param_array[i, 0]),
                ser=ConstantRate(rate=param_array[i, 1]),
                perform_param=perform_param)

        elif isinstance(arrival, MMOO):
            setting = SingleServerPerform(
                arr=MMOO(
                    mu=param_array[i, 0],
                    lamb=param_array[i, 1],
                    burst=param_array[i, 2]),
                ser=ConstantRate(rate=param_array[i, 3]),
                perform_param=perform_param)

        else:
            raise NameError("Arrival parameter " + arrival.__class__.__name__ +
                            " is infeasible")

            # standard_bound, new_bound = compute_improvement()
        res_array[i, 0], res_array[i, 1] = compute_improvement(
            setting=setting, opt_method=opt_method)

        # This might be a very dangerous condition
        if res_array[i, 1] == inf:
            res_array[i, ] = nan

        if i % floor(total_iterations / 10) == 0:
            print("iteration {0} of {1}".format(i, total_iterations))

    res_dict = data_array_to_results(
        arrival=arrival,
        service=service,
        metric=metric,
        param_array=param_array,
        res_array=res_array,
        number_servers=1)

    res_dict.update({
        "delta_time": perform_param.value,
        "optimization": opt_method.name,
        "metric": metric,
        "iterations": total_iterations,
        "MCDistribution": mc_dist.mc_dist_name.name,
        "MCParam": mc_dist.param_to_string()
    })

    with open(
            "single_" + perform_param.perform_metric.name + "_" +
            arrival.__class__.__name__ + "_results_MC" +
            mc_dist.mc_dist_name.name + "_" + opt_method.name + "_" + metric +
            ".csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in res_dict.items():
            writer.writerow([key, value])

    return res_dict


def grid_param_single_exp(perform_param: PerformParameter,
                          opt_method: OptMethod, metric: str, lamb1_range,
                          rate1_range) -> dict:
    """Choose parameters along a grid"""

    total_iterations = len(lamb1_range) * len(rate1_range)

    param_array = np.empty([total_iterations, 2])
    res_array = np.empty([total_iterations, 2])

    i = 0
    for lamb1 in lamb1_range:
        for rate1 in rate1_range:
            setting = SingleServerPerform(
                arr=ExponentialArrival(lamb=lamb1),
                ser=ConstantRate(rate=rate1),
                perform_param=perform_param)
            param_array[i, 0] = lamb1
            param_array[i, 1] = rate1

            res_array[i, 0], res_array[i, 1] = compute_improvement(
                setting=setting, opt_method=opt_method, number_l=1)
            if res_array[i, 1] == inf:
                res_array[i, ] = nan

            i += 1
            if i % floor(total_iterations / 10) == 0:
                print("iteration {0} of {1}".format(i, total_iterations))

    exp_arrival = ExponentialArrival(lamb=1)
    const_service = ConstantRate(rate=1)

    return data_array_to_results(
        arrival=exp_arrival,
        service=const_service,
        metric=metric,
        param_array=param_array,
        res_array=param_array,
        number_servers=1)


if __name__ == '__main__':
    REPETITIONS = 10**3

    OUTPUT_TIME4 = PerformParameter(
        perform_metric=PerformMetric.OUTPUT, value=4)

    EXP_ARRIVAL = ExponentialArrival()
    MMOO_ARRIVAL = MMOO()
    CONST_RATE = ConstantRate()

    COMMON_OPTIMIZATION = OptMethod.GRID_SEARCH

    MC_UNIF20 = MonteCarloDist(mc_dist_name=MCName.UNIFORM, param_list=[20.0])
    MC_EXP1 = MonteCarloDist(MCName.EXPONENTIAL, [1.0])

    print(
        csv_single_server_param(
            arrival=EXP_ARRIVAL,
            service=CONST_RATE,
            perform_param=OUTPUT_TIME4,
            opt_method=COMMON_OPTIMIZATION,
            total_iterations=REPETITIONS,
            mc_dist=MC_UNIF20))
    print(
        csv_single_server_param(
            arrival=MMOO_ARRIVAL,
            service=CONST_RATE,
            perform_param=OUTPUT_TIME4,
            opt_method=COMMON_OPTIMIZATION,
            total_iterations=REPETITIONS,
            mc_dist=MC_UNIF20))

    def fun1():
        """Auxiliary function for parallelization"""
        print(
            csv_single_server_param(
                arrival=EXP_ARRIVAL,
                service=CONST_RATE,
                perform_param=OUTPUT_TIME4,
                opt_method=COMMON_OPTIMIZATION,
                total_iterations=REPETITIONS,
                mc_dist=MC_EXP1))

    def fun2():
        """Auxiliary function for parallelization"""
        print(
            csv_single_server_param(
                arrival=MMOO_ARRIVAL,
                service=CONST_RATE,
                perform_param=OUTPUT_TIME4,
                opt_method=COMMON_OPTIMIZATION,
                total_iterations=REPETITIONS,
                mc_dist=MC_EXP1))

    def run_in_parallel(*funcs):
        """Runs auxiliary functions in parallel"""
        proc = []
        for func in funcs:
            process_instance = Process(target=func)
            process_instance.start()
            proc.append(process_instance)
        for process_instance in proc:
            process_instance.join()

    run_in_parallel(fun1, fun2)

    # print(
    #     grid_param_single_exp(
    #         perform_param=OUTPUT_TIME,
    #         opt_method=OptMethod.GRID_SEARCH,
    #         lamb1_range=[0.1, 0.3, 0.5, 0.7, 1, 2, 4, 8, 12],
    #         rate1_range=[0.1, 0.3, 0.5, 0.7, 1, 2, 4, 8, 12]))
