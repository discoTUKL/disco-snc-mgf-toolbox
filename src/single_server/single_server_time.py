"""Compute average computation time for different parameters"""

from math import floor

import numpy as np

from library.array_to_results import time_array_to_results
from library.compare_old_new import compute_overhead
from library.mc_enum import MCEnum
from library.monte_carlo_dist import MonteCarloDist
from library.perform_parameter import PerformParameter
from nc_operations.perform_enum import PerformEnum
from nc_processes.arrival_distribution import DM1, MMOO
from nc_processes.arrival_enum import ArrivalEnum
from nc_processes.constant_rate_server import ConstantRate
from optimization.opt_method import OptMethod
from single_server.single_server_perform import SingleServerPerform


def mc_time_single(arrival_enum: ArrivalEnum,
                   perform_param: PerformParameter, opt_method: OptMethod,
                   mc_dist: MonteCarloDist) -> dict:
    """Chooses parameters by Monte Carlo type random choice"""
    total_iterations = 10**4

    time_ratio = {"Number_of_servers": "Ratio"}

    # 1 Parameter for service
    size_array = [total_iterations, arrival_enum.number_parameters() + 1]
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

    time_array = np.empty([total_iterations, 2])

    for i in range(total_iterations):
        if arrival_enum == ArrivalEnum.DM1:
            setting = SingleServerPerform(
                arr=DM1(lamb=param_array[i, 0]),
                const_rate=ConstantRate(rate=param_array[i, 1]),
                perform_param=perform_param)

        elif arrival_enum == ArrivalEnum.MMOO:
            setting = SingleServerPerform(
                arr=MMOO(
                    mu=param_array[i, 0],
                    lamb=param_array[i, 1],
                    burst=param_array[i, 2]),
                const_rate=ConstantRate(rate=param_array[i, 3]),
                perform_param=perform_param)

        else:
            raise NameError("Arrival parameter {0} is infeasible".format(
                arrival_enum.name))

        # time_standard, time_lyapunov = compute_overhead()
        time_array[i, 0], time_array[i, 1] = compute_overhead(
            setting=setting, opt_method=opt_method, number_l=1)

        if i % floor(total_iterations / 10) == 0:
            print("iteration {0} of {1}".format(i, total_iterations))

    return time_array_to_results(
        arrival_enum=arrival_enum,
        time_array=time_array,
        number_servers=1,
        time_ratio=time_ratio)


if __name__ == '__main__':
    OUTPUT_TIME = PerformParameter(perform_metric=PerformEnum.OUTPUT, value=4)

    COMMON_OPTIMIZATION = OptMethod.GRID_SEARCH

    MC_UNIF20 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[20.0])

    print(
        mc_time_single(
            arrival_enum=ArrivalEnum.DM1,
            perform_param=OUTPUT_TIME,
            opt_method=COMMON_OPTIMIZATION,
            mc_dist=MC_UNIF20))

    print(
        mc_time_single(
            arrival_enum=ArrivalEnum.MMOO,
            perform_param=OUTPUT_TIME,
            opt_method=COMMON_OPTIMIZATION,
            mc_dist=MC_UNIF20))
