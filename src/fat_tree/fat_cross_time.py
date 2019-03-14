"""Compute average computation time for different parameters."""

import csv
from math import floor
from typing import List

import numpy as np
from tqdm import tqdm  # Progressbar in for loop

from bound_evaluation.array_to_results import time_array_to_results
from bound_evaluation.mc_enum import MCEnum
from bound_evaluation.mc_enum_to_dist import mc_enum_to_dist
from bound_evaluation.monte_carlo_dist import MonteCarloDist
from fat_tree.fat_cross_perform import FatCrossPerform
from h_mitigator.compare_mitigator import compare_time
from nc_arrivals.arrival_enum import ArrivalEnum
from nc_arrivals.markov_modulated import MMOOFluid
from nc_arrivals.qt import DM1
from nc_operations.perform_enum import PerformEnum
from nc_service.constant_rate_server import ConstantRate
from optimization.opt_method import OptMethod
from utils.perform_parameter import PerformParameter

########################################################################
# Find Optimal Parameters
########################################################################


def mc_time_fat_cross(arrival_enum: ArrivalEnum,
                      list_number_servers: List[int],
                      perform_param: PerformParameter, opt_method: OptMethod,
                      mc_dist: MonteCarloDist) -> dict:
    """Chooses parameters by Monte Carlo type random choice."""
    total_iterations = 10**4

    time_ratio = {"Number_of_servers": "Ratio"}

    for num_serv in tqdm(list_number_servers):

        # 1 Parameter for service

        size_array = [
            total_iterations, (arrival_enum.number_parameters() + 1) * num_serv
        ]
        # [rows, columns]

        param_array = mc_enum_to_dist(mc_dist=mc_dist, size=size_array)

        time_array = np.empty([total_iterations, 2])

        for i in range(total_iterations):
            if arrival_enum == ArrivalEnum.DM1:
                arrive_list = [
                    DM1(lamb=param_array[i, j]) for j in range(num_serv)
                ]
            elif arrival_enum == ArrivalEnum.MMOO:
                arrive_list = [
                    MMOOFluid(
                        mu=param_array[i, j],
                        lamb=param_array[i, num_serv + j],
                        burst=param_array[i, 2 * num_serv + j])
                    for j in range(num_serv)
                ]

            else:
                raise NameError("Arrival parameter {0} is infeasible".format(
                    arrival_enum.name))

            service_list = [
                ConstantRate(rate=param_array[
                    i, arrival_enum.number_parameters() * num_serv + j])
                for j in range(num_serv)
            ]

            setting = FatCrossPerform(
                arr_list=arrive_list,
                ser_list=service_list,
                perform_param=perform_param)

            # time_standard, time_lyapunov = compare_time()
            time_array[i, 0], time_array[i, 1] = compare_time(
                setting=setting, opt_method=opt_method, number_l=num_serv - 1)

            if i % floor(total_iterations / 10) == 0:
                print("iteration {0} of {1}".format(i, total_iterations))

        print(
            time_array_to_results(
                arrival_enum=arrival_enum,
                time_array=time_array,
                number_servers=num_serv,
                time_ratio=time_ratio))

    with open(
        (f"time_{perform_param.__str__()}_{arrival_enum.name}_{opt_method.name}.csv"
         ), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in time_ratio.items():
            writer.writerow([key, value])

    return time_ratio


if __name__ == '__main__':
    DELAY_PROB = PerformParameter(
        perform_metric=PerformEnum.DELAY_PROB, value=4)

    COMMON_OPTIMIZATION = OptMethod.GRID_SEARCH

    MC_UNIF20 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[20.0])

    list_number_servers1 = [2, 4, 6, 8, 10, 12]

    print(
        mc_time_fat_cross(
            arrival_enum=ArrivalEnum.DM1,
            list_number_servers=list_number_servers1,
            perform_param=DELAY_PROB,
            opt_method=COMMON_OPTIMIZATION,
            mc_dist=MC_UNIF20))

    print(
        mc_time_fat_cross(
            arrival_enum=ArrivalEnum.MMOO,
            list_number_servers=list_number_servers1,
            perform_param=DELAY_PROB,
            opt_method=COMMON_OPTIMIZATION,
            mc_dist=MC_UNIF20))
