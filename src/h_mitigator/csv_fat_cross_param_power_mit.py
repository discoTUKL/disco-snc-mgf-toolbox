"""Compute optimal and average improvement for different parameters."""

import csv

import numpy as np
from tqdm import tqdm

from bound_evaluation.array_to_results import two_col_array_to_results
from bound_evaluation.change_enum import ChangeEnum
from bound_evaluation.mc_enum import MCEnum
from bound_evaluation.mc_enum_to_dist import mc_enum_to_dist
from bound_evaluation.monte_carlo_dist import MonteCarloDist
from h_mitigator.compare_mitigator import compare_mitigator
from h_mitigator.fat_cross_perform import FatCrossPerform
from nc_arrivals.arrival_enum import ArrivalEnum
from nc_arrivals.ebb import EBB
from nc_arrivals.markov_modulated import MMOOFluid
from nc_arrivals.qt import DM1, MD1
from nc_arrivals.regulated_arrivals import (LeakyBucketMassOne,
                                            TokenBucketConstant)
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.opt_method import OptMethod
from utils.perform_parameter import PerformParameter

########################################################################
# Find Optimal Parameters
########################################################################


def csv_fat_cross_param_power(arrival_enum: ArrivalEnum, number_servers: int,
                              perform_param: PerformParameter,
                              opt_method: OptMethod, mc_dist: MonteCarloDist,
                              total_iterations: int,
                              target_util: float) -> dict:
    """Chooses parameters by Monte Carlo type random choice."""
    valid_iterations = total_iterations
    compare_metric = ChangeEnum.RATIO_REF_NEW

    size_array = [
        total_iterations,
        (arrival_enum.number_parameters() + 1) * number_servers
        # const_rate has 1 parameter
    ]
    # [rows, columns]

    param_array = mc_enum_to_dist(mc_dist=mc_dist, size=size_array)

    res_array = np.empty([total_iterations, 2])

    # print(res_array)

    for i in tqdm(range(total_iterations), total=total_iterations):
        if arrival_enum == ArrivalEnum.DM1:
            arr_list = [
                DM1(lamb=param_array[i, j]) for j in range(number_servers)
            ]

        elif arrival_enum == ArrivalEnum.MD1:
            arr_list = [
                MD1(lamb=param_array[i, j], mu=1.0)
                for j in range(number_servers)
            ]

        elif arrival_enum == ArrivalEnum.MMOOFluid:
            arr_list = [
                MMOOFluid(mu=param_array[i, j],
                          lamb=param_array[i, number_servers + j],
                          burst=param_array[i, 2 * number_servers + j])
                for j in range(number_servers)
            ]

        elif arrival_enum == ArrivalEnum.EBB:
            arr_list = [
                EBB(factor_m=param_array[i, j],
                    decay=param_array[i, number_servers + j],
                    rho_single=param_array[i, 2 * number_servers + j])
                for j in range(number_servers)
            ]

        elif arrival_enum == ArrivalEnum.MassOne:
            arr_list = [
                LeakyBucketMassOne(sigma_single=param_array[i, j],
                                   rho_single=param_array[i, number_servers +
                                                          j],
                                   n=20) for j in range(number_servers)
            ]
            # TODO: note that n is fixed

        elif arrival_enum == ArrivalEnum.TBConst:
            arr_list = [
                TokenBucketConstant(sigma_single=param_array[i, j],
                                    rho_single=param_array[i, number_servers +
                                                           j],
                                    n=1) for j in range(number_servers)
            ]

        else:
            raise NotImplementedError(f"Arrival parameter {arrival_enum.name} "
                                      f"is infeasible")

        ser_list = [
            ConstantRateServer(rate=param_array[
                i, arrival_enum.number_parameters() * number_servers + j])
            for j in range(number_servers)
        ]

        fat_cross_setting = FatCrossPerform(arr_list=arr_list,
                                            ser_list=ser_list,
                                            perform_param=perform_param)

        computation_necessary = True

        # print(res_array[i, ])
        if target_util > 0.0:
            util = fat_cross_setting.approximate_utilization()
            if util < target_util or util > 1:
                res_array[i, ] = np.nan
                computation_necessary = False

        if computation_necessary:
            # standard_bound, h_mit_bound = compare_mitigator()
            res_array[i, 0], res_array[i, 1] = compare_mitigator(
                setting=fat_cross_setting,
                opt_method=opt_method,
                number_l=number_servers - 1)

            if (perform_param.perform_metric == PerformEnum.DELAY_PROB
                    and res_array[i, 1] > 1.0):
                # write as nan if second (in particular both) value(s) are > 1.0
                res_array[i, ] = np.nan

        if np.isnan(res_array[i, 0]) or np.isnan(res_array[i, 1]):
            res_array[i, ] = np.nan
            valid_iterations -= 1

    res_dict = two_col_array_to_results(arrival_enum=arrival_enum,
                                        param_array=param_array,
                                        res_array=res_array,
                                        number_servers=number_servers,
                                        valid_iterations=valid_iterations,
                                        compare_metric=compare_metric)

    res_dict.update({
        "iterations": total_iterations,
        "T": perform_param.value,
        "optimization": opt_method.name,
        "compare_metric": compare_metric.name,
        "MCDistribution": mc_dist.to_name(),
        "MCParam": mc_dist.param_to_string(),
        "number_servers": number_servers
    })

    with open(
            f"sim_{perform_param.to_name()}_{arrival_enum.name}_results_"
            f"MC{mc_dist.to_name()}_{opt_method.name}"
            f"_{compare_metric.name}.csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in res_dict.items():
            writer.writerow([key, value])

    return res_dict


if __name__ == '__main__':
    DELAY_PROB10 = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                    value=10)

    COMMON_OPTIMIZATION = OptMethod.GRID_SEARCH

    # MC_UNIF20 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[20.0])
    MC_UNIF10 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[10.0])
    MC_EXP1 = MonteCarloDist(mc_enum=MCEnum.EXPONENTIAL, param_list=[1.0])

    ARRIVAL_PROCESSES = [
        ArrivalEnum.DM1, ArrivalEnum.MMOOFluid, ArrivalEnum.MD1
    ]

    for PROCESS in ARRIVAL_PROCESSES:
        print(
            csv_fat_cross_param_power(arrival_enum=PROCESS,
                                      number_servers=2,
                                      perform_param=DELAY_PROB10,
                                      opt_method=COMMON_OPTIMIZATION,
                                      mc_dist=MC_EXP1,
                                      total_iterations=10**5,
                                      target_util=0.5))

        print(
            csv_fat_cross_param_power(arrival_enum=PROCESS,
                                      number_servers=2,
                                      perform_param=DELAY_PROB10,
                                      opt_method=COMMON_OPTIMIZATION,
                                      mc_dist=MC_UNIF10,
                                      total_iterations=10**5,
                                      target_util=0.5))
