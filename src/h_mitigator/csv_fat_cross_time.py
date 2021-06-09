"""Compute average computation time for different parameters."""

import csv
from typing import List

import numpy as np
from tqdm import tqdm  # Progressbar in for loop

from bound_evaluation.mc_enum import MCEnum
from bound_evaluation.mc_enum_to_dist import mc_enum_to_dist
from bound_evaluation.monte_carlo_dist import MonteCarloDist
from h_mitigator.array_to_results import time_array_to_results
from h_mitigator.compare_mitigator import compare_time
from h_mitigator.fat_cross_perform import FatCrossPerform
from nc_arrivals.arrival_enum import ArrivalEnum
from nc_arrivals.markov_modulated import MMOOFluid
from nc_arrivals.iid import DM1
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.opt_method import OptMethod
from utils.perform_parameter import PerformParameter

########################################################################
# Find Optimal Parameters
########################################################################


def csv_fat_cross_time(arrival_enum: ArrivalEnum,
                       list_number_servers: List[int],
                       perform_param: PerformParameter, opt_method: OptMethod,
                       mc_dist: MonteCarloDist, target_util: float) -> dict:
    """Chooses parameters by Monte Carlo type random choice."""
    total_iterations = 10**5

    time_ratio = {"Number_of_servers": "Ratio"}

    for number_servers in list_number_servers:
        print(f"number of servers = {number_servers}")
        # 1 Parameter for service

        param_array = mc_enum_to_dist(arrival_enum=arrival_enum,
                                      mc_dist=mc_dist,
                                      number_flows=number_servers,
                                      number_servers=number_servers,
                                      total_iterations=total_iterations)

        time_array = np.empty([total_iterations, 2])

        for i in tqdm(range(total_iterations)):
            if arrival_enum == ArrivalEnum.DM1:
                arrive_list = [
                    DM1(lamb=param_array[i, j]) for j in range(number_servers)
                ]
            elif arrival_enum == ArrivalEnum.MMOOFluid:
                arrive_list = [
                    MMOOFluid(mu=param_array[i, j],
                              lamb=param_array[i, number_servers + j],
                              peak_rate=param_array[i, 2 * number_servers + j])
                    for j in range(number_servers)
                ]

            else:
                raise NameError(f"Arrival parameter {arrival_enum.name} "
                                f"is infeasible")

            service_list = [
                ConstantRateServer(
                    rate=param_array[i,
                                     arrival_enum.number_parameters() *
                                     number_servers + j])
                for j in range(number_servers)
            ]

            fat_cross_setting = FatCrossPerform(arr_list=arrive_list,
                                                ser_list=service_list,
                                                perform_param=perform_param)

            computation_necessary = True

            # print(res_array[i, ])
            if target_util > 0.0:
                util = fat_cross_setting.approximate_utilization()
                if util < target_util or util > 1:
                    time_array[i, ] = np.nan
                    computation_necessary = False

            if computation_necessary:
                # time_standard, time_lyapunov = compare_time()
                time_array[i, 0], time_array[i, 1] = compare_time(
                    setting=fat_cross_setting,
                    opt_method=opt_method,
                    number_l=number_servers - 1)

        print(
            time_array_to_results(arrival_enum=arrival_enum,
                                  time_array=time_array,
                                  number_servers=number_servers,
                                  time_ratio=time_ratio))

    filename = (f"time_{perform_param.to_name()}_{arrival_enum.name}"
                f"_{opt_method.name}.csv")

    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in time_ratio.items():
            writer.writerow([key, value])

    return time_ratio


if __name__ == '__main__':
    DELAY_PROB = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                  value=4)

    COMMON_OPTIMIZATION = OptMethod.PATTERN_SEARCH

    MC_UNIF20 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[20.0])

    list_number_servers1 = [2, 4, 6, 8, 10, 12]

    print(
        csv_fat_cross_time(arrival_enum=ArrivalEnum.DM1,
                           list_number_servers=list_number_servers1,
                           perform_param=DELAY_PROB,
                           opt_method=COMMON_OPTIMIZATION,
                           mc_dist=MC_UNIF20,
                           target_util=0.5))

    print(
        csv_fat_cross_time(arrival_enum=ArrivalEnum.MMOOFluid,
                           list_number_servers=list_number_servers1,
                           perform_param=DELAY_PROB,
                           opt_method=COMMON_OPTIMIZATION,
                           mc_dist=MC_UNIF20,
                           target_util=0.5))
