"""Compute average computation time for different parameters"""

from math import floor

import numpy as np

from library.array_to_results import time_array_to_results
from library.compare_old_new import compute_overhead
from library.mc_name import MCName
from library.monte_carlo_dist import MonteCarloDist
from library.perform_parameter import PerformParameter
from nc_operations.perform_metric import PerformMetric
from nc_processes.arrival_distribution import (MMOO, ArrivalDistribution,
                                               ExponentialArrival)
from nc_processes.service import ConstantRate
from optimization.opt_method import OptMethod
from single_server.single_server_perform import SingleServerPerform


def mc_time_single(arrival: ArrivalDistribution,
                   perform_param: PerformParameter, opt_method: OptMethod,
                   total_iterations: int, mc_dist: MonteCarloDist) -> dict:
    """Chooses parameters by Monte Carlo type random choice"""

    time_ratio = {"Number_of_servers": "Ratio"}

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
        raise NameError("Distribution parameter {0} is infeasible".format(
            mc_dist.mc_dist_name))

    time_array = np.empty([total_iterations, 2])

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
            raise NameError("Arrival parameter {0} is infeasible".format(
                arrival.__class__.__name__))

        # time_standard, time_lyapunov = compute_overhead()
        time_array[i, 0], time_array[i, 1] = compute_overhead(
            setting=setting, opt_method=opt_method, number_l=1)

        if i % floor(total_iterations / 10) == 0:
            print("iteration {0} of {1}".format(i, total_iterations))

    return time_array_to_results(
        arrival=arrival,
        time_array=time_array,
        number_servers=1,
        time_ratio=time_ratio)


if __name__ == '__main__':
    REPETITIONS = 10**4

    OUTPUT_TIME = PerformParameter(
        perform_metric=PerformMetric.OUTPUT, value=4)

    EXP_ARRIVAL1 = ExponentialArrival()
    MMOO_ARRIVAL1 = MMOO()

    COMMON_OPTIMIZATION = OptMethod.GRID_SEARCH

    MC_UNIF20 = MonteCarloDist(mc_dist_name=MCName.UNIFORM, param_list=[20.0])

    print(
        mc_time_single(
            arrival=EXP_ARRIVAL1,
            perform_param=OUTPUT_TIME,
            opt_method=COMMON_OPTIMIZATION,
            total_iterations=REPETITIONS,
            mc_dist=MC_UNIF20))

    print(
        mc_time_single(
            arrival=MMOO_ARRIVAL1,
            perform_param=OUTPUT_TIME,
            opt_method=COMMON_OPTIMIZATION,
            total_iterations=REPETITIONS,
            mc_dist=MC_UNIF20))
