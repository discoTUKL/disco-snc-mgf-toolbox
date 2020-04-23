"""Compute optimal and average improvement for different parameters."""

import csv

import numpy as np
from tqdm import tqdm

from bound_evaluation.change_enum import ChangeEnum
from bound_evaluation.mc_enum import MCEnum
from bound_evaluation.mc_enum_to_dist import mc_enum_to_dist
from bound_evaluation.monte_carlo_dist import MonteCarloDist
from msob_and_fp.compare_avoid_dep import compare_time_211, compare_time_212
from msob_and_fp.msob_fp_array_to_results import time_array_to_results
from msob_and_fp.overlapping_tandem_perform import OverlappingTandemPerform
from msob_and_fp.square_perform import SquarePerform
from nc_arrivals.arrival_enum import ArrivalEnum
from nc_arrivals.markov_modulated import MMOODisc, MMOOFluid
from nc_arrivals.qt import DM1, MD1
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.opt_method import OptMethod
from utils.perform_parameter import PerformParameter

########################################################################
# Find Optimal Parameters
########################################################################


def csv_msob_fp_time(name: str, number_flows: int, number_servers: int,
                     arrival_enum: ArrivalEnum,
                     perform_param: PerformParameter, opt_method: OptMethod,
                     mc_dist: MonteCarloDist, comparator: callable,
                     total_iterations: int, target_util: float) -> dict:
    """Chooses parameters by Monte Carlo type random choice."""

    param_array = mc_enum_to_dist(arrival_enum=arrival_enum,
                                  mc_dist=mc_dist,
                                  number_flows=number_flows,
                                  number_servers=number_servers,
                                  total_iterations=total_iterations)

    time_array = np.empty([total_iterations, 3])
    # 3 approaches to compare

    for i in tqdm(range(total_iterations), total=total_iterations):
        if arrival_enum == ArrivalEnum.DM1:
            arr_list = [
                DM1(lamb=param_array[i, j]) for j in range(number_flows)
            ]

        elif arrival_enum == ArrivalEnum.MD1:
            arr_list = [
                MD1(lamb=param_array[i, j], mu=1.0)
                for j in range(number_flows)
            ]

        elif arrival_enum == ArrivalEnum.MMOODisc:
            arr_list = [
                MMOODisc(stay_on=param_array[i, j],
                         stay_off=param_array[i, number_flows + j],
                         peak_rate=param_array[i, 2 * number_flows + j])
                for j in range(number_flows)
            ]

        elif arrival_enum == ArrivalEnum.MMOOFluid:
            arr_list = [
                MMOOFluid(mu=param_array[i, j],
                          lamb=param_array[i, number_flows + j],
                          peak_rate=param_array[i, 2 * number_flows + j])
                for j in range(number_flows)
            ]

        else:
            raise NotImplementedError(f"Arrival parameter {arrival_enum.name} "
                                      f"is infeasible")

        ser_list = [
            ConstantRateServer(
                rate=param_array[i,
                                 arrival_enum.number_parameters() *
                                 number_flows + j])
            for j in range(number_servers)
        ]

        if name == "overlapping_tandem":
            setting = OverlappingTandemPerform(arr_list=arr_list,
                                               ser_list=ser_list,
                                               perform_param=perform_param)
        elif name == "square":
            setting = SquarePerform(arr_list=arr_list,
                                    ser_list=ser_list,
                                    perform_param=perform_param)
        else:
            raise NotImplementedError("this topology is not implemented")

        computation_necessary = True

        if target_util > 0.0:
            util = setting.approximate_utilization()
            if util < target_util or util > 1:
                time_array[i, ] = np.nan
                computation_necessary = False

        if computation_necessary:
            # standard_bound, server_bound, fp_bound = compare_avoid_dep()
            time_array[i, 0], time_array[i, 1], time_array[i, 2] = comparator(
                setting=setting)

            if (perform_param.perform_metric == PerformEnum.DELAY_PROB
                    and np.nanmin(time_array[i, ]) > 1.0):
                # np.nanmin(time_array[i, ]) is the smallest value
                time_array[i, ] = np.nan

    time_dict = time_array_to_results(title=name, time_array=time_array)

    time_dict.update({
        "iterations": total_iterations,
        "T": perform_param.value,
        "optimization": opt_method.name,
        "MCDistribution": mc_dist.to_name(),
        "MCParam": mc_dist.param_to_string()
    })

    filename = name
    filename += f"_time_{perform_param.to_name()}_{arrival_enum.name}" \
        f"_MC{mc_dist.to_name()}_{opt_method.name}_util_{target_util}"

    with open(filename + ".csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in time_dict.items():
            writer.writerow([key, value])

    return time_dict


if __name__ == '__main__':
    # DELAY_PROB10 = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
    #                                 value=10)
    DELAY_6 = PerformParameter(perform_metric=PerformEnum.DELAY,
                               value=10**(-6))

    COMMON_PERFORM_PARAM = DELAY_6
    COMMON_OPTIMIZATION = OptMethod.GRID_SEARCH
    COMMON_METRIC = ChangeEnum.RATIO_REF_NEW
    TARGET_UTIL = 0.75

    # MC_UNIF20 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[20.0])
    MC_UNIF10 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[10.0])

    PROCESS = ArrivalEnum.MD1

    print(
        csv_msob_fp_time(name="overlapping_tandem",
                         number_flows=3,
                         number_servers=3,
                         arrival_enum=PROCESS,
                         perform_param=COMMON_PERFORM_PARAM,
                         opt_method=COMMON_OPTIMIZATION,
                         mc_dist=MC_UNIF10,
                         comparator=compare_time_211,
                         total_iterations=10**4,
                         target_util=TARGET_UTIL))

    print(
        csv_msob_fp_time(name="square",
                         number_flows=4,
                         number_servers=4,
                         arrival_enum=PROCESS,
                         perform_param=COMMON_PERFORM_PARAM,
                         opt_method=COMMON_OPTIMIZATION,
                         mc_dist=MC_UNIF10,
                         comparator=compare_time_212,
                         total_iterations=10**4,
                         target_util=TARGET_UTIL))
