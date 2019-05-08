"""Compute average computation time for different parameters"""

from math import floor

import numpy as np

from bound_evaluation.array_to_results import time_array_to_results
from bound_evaluation.mc_enum import MCEnum
from bound_evaluation.mc_enum_to_dist import mc_enum_to_dist
from bound_evaluation.monte_carlo_dist import MonteCarloDist
from h_mitigator.compare_mitigator import compare_time
from h_mitigator.single_server_mit_perform import SingleServerMitPerform
from nc_arrivals.arrival_enum import ArrivalEnum
from nc_arrivals.markov_modulated import MMOOFluid
from nc_arrivals.qt import DM1
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.opt_method import OptMethod
from utils.perform_parameter import PerformParameter


def mc_time_single(arrival_enum: ArrivalEnum, perform_param: PerformParameter,
                   opt_method: OptMethod, mc_dist: MonteCarloDist) -> dict:
    """Chooses parameters by Monte Carlo type random choice"""
    total_iterations = 10**4

    time_ratio = {"Number_of_servers": "Ratio"}

    # 1 Parameter for service
    size_array = [total_iterations, arrival_enum.number_parameters() + 1]
    # [rows, columns]

    param_array = mc_enum_to_dist(mc_dist=mc_dist, size=size_array)

    time_array = np.empty([total_iterations, 2])

    for i in range(total_iterations):
        if arrival_enum == ArrivalEnum.DM1:
            setting = SingleServerMitPerform(
                arr=DM1(lamb=param_array[i, 0]),
                const_rate=ConstantRateServer(rate=param_array[i, 1]),
                perform_param=perform_param)

        elif arrival_enum == ArrivalEnum.MMOOFluid:
            setting = SingleServerMitPerform(
                arr=MMOOFluid(
                    mu=param_array[i, 0],
                    lamb=param_array[i, 1],
                    burst=param_array[i, 2]),
                const_rate=ConstantRateServer(rate=param_array[i, 3]),
                perform_param=perform_param)

        else:
            raise NameError(
                f"Arrival parameter {arrival_enum.name} is infeasible")

        # time_standard, time_lyapunov = compare_time()
        time_array[i, 0], time_array[i, 1] = compare_time(
            setting=setting, opt_method=opt_method, number_l=1)

        if i % floor(total_iterations / 10) == 0:
            print(f"iteration {i} of {total_iterations}")

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
            arrival_enum=ArrivalEnum.MMOOFluid,
            perform_param=OUTPUT_TIME,
            opt_method=COMMON_OPTIMIZATION,
            mc_dist=MC_UNIF20))
