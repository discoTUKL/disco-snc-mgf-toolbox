"""Compute optimal and average improvement for different parameters."""

import csv
from warnings import warn

import numpy as np
from bound_evaluation.change_enum import ChangeEnum
from bound_evaluation.manipulate_data import remove_full_nan_rows
from bound_evaluation.mc_enum import MCEnum
from bound_evaluation.mc_enum_to_dist import mc_enum_to_dist
from bound_evaluation.monte_carlo_dist import MonteCarloDist
from nc_arrivals.arrival_enum import ArrivalEnum
from nc_arrivals.iid import DM1, MD1, DGamma1, DWeibull1
from nc_arrivals.markov_modulated import MMOODisc, MMOOCont
from nc_arrivals.regulated_arrivals import (DetermTokenBucket,
                                            LeakyBucketMassoulie)
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.opt_method import OptMethod
from tqdm import tqdm
from utils.exceptions import NotEnoughResults
from utils.perform_parameter import PerformParameter

from h_mitigator.array_to_results import two_col_array_to_results
from h_mitigator.compare_mitigator import compare_mitigator
from h_mitigator.fat_cross_perform import FatCrossPerform


def csv_fat_cross_param_power(name: str, arrival_enum: ArrivalEnum,
                              number_flows: int, number_servers: int,
                              perform_param: PerformParameter,
                              opt_method: OptMethod, mc_dist: MonteCarloDist,
                              compare_metric: ChangeEnum,
                              total_iterations: int,
                              target_util: float) -> dict:
    """Chooses parameters by Monte Carlo type random choice."""
    param_array = mc_enum_to_dist(arrival_enum=arrival_enum,
                                  mc_dist=mc_dist,
                                  number_flows=number_flows,
                                  number_servers=number_servers,
                                  total_iterations=total_iterations)

    res_array = np.empty([total_iterations, 2])

    # print(res_array)

    for i in tqdm(range(total_iterations), total=total_iterations):
        if arrival_enum == ArrivalEnum.DM1:
            arr_list = [
                DM1(lamb=param_array[i, j]) for j in range(number_flows)
            ]

        elif arrival_enum == ArrivalEnum.DGamma1:
            arr_list = [
                DGamma1(alpha_shape=param_array[i, j],
                        beta_rate=param_array[i, number_flows + j])
                for j in range(number_flows)
            ]

        elif arrival_enum == ArrivalEnum.DWeibull1:
            arr_list = [
                DWeibull1(lamb=param_array[i, j]) for j in range(number_flows)
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
                MMOOCont(mu=param_array[i, j],
                         lamb=param_array[i, number_flows + j],
                         peak_rate=param_array[i, 2 * number_flows + j])
                for j in range(number_flows)
            ]

        elif arrival_enum == ArrivalEnum.Massoulie:
            arr_list = [
                LeakyBucketMassoulie(sigma_single=param_array[i, j],
                                     rho_single=param_array[i,
                                                            number_flows + j],
                                     n=20) for j in range(number_flows)
            ]
            # NOTE: n is fixed

        elif arrival_enum == ArrivalEnum.TBConst:
            arr_list = [
                DetermTokenBucket(sigma_single=param_array[i, j],
                                  rho_single=param_array[i, number_flows + j],
                                  n=1) for j in range(number_flows)
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

        fat_cross_setting = FatCrossPerform(arr_list=arr_list,
                                            ser_list=ser_list,
                                            perform_param=perform_param)

        computation_necessary = True

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

    res_array_no_full_nan = remove_full_nan_rows(full_array=res_array)
    valid_iterations = res_array_no_full_nan.shape[0]

    if valid_iterations < total_iterations * 0.2:
        warn(f"Many nan's: {total_iterations - valid_iterations} nans "
             f"out of {total_iterations}!")

        if valid_iterations < 100:
            raise NotEnoughResults("result is useless")

    res_dict = two_col_array_to_results(arrival_enum=arrival_enum,
                                        param_array=param_array,
                                        res_array=res_array,
                                        number_servers=number_servers,
                                        compare_metric=compare_metric)

    res_dict.update({
        "iterations": total_iterations,
        "PerformParamValue": perform_param.value,
        "optimization": opt_method.name,
        "compare_metric": compare_metric.name,
        "MCDistribution": mc_dist.to_name(),
        "MCParam": mc_dist.param_to_string(),
        "number_servers": number_servers
    })

    filename = name
    filename += f"_results_{perform_param.to_name()}_{arrival_enum.name}_" \
                f"MC{mc_dist.to_name()}_{opt_method.name}_" \
                f"{compare_metric.name}_util_{target_util}"

    with open(filename + ".csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in res_dict.items():
            writer.writerow([key, value])

    return res_dict


if __name__ == '__main__':
    COMMON_PERFORM_PARAM = PerformParameter(
        perform_metric=PerformEnum.DELAY_PROB, value=10)

    # COMMON_PERFORM_PARAM = PerformParameter(perform_metric=PerformEnum.DELAY,
    #                                         value=1e-6)

    COMMON_OPTIMIZATION = OptMethod.GRID_SEARCH
    COMMON_METRIC = ChangeEnum.RATIO_REF_NEW
    TARGET_UTIL = 0.7

    # MC_UNIF20 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[20.0])
    MC_UNIF10 = MonteCarloDist(mc_enum=MCEnum.UNIFORM, param_list=[10.0])
    MC_EXP1 = MonteCarloDist(mc_enum=MCEnum.EXPONENTIAL, param_list=[1.0])

    # ARRIVAL_PROCESSES = [
    #     ArrivalEnum.DM1, ArrivalEnum.MMOOFluid, ArrivalEnum.MD1
    # ]

    ARRIVAL_PROCESSES = [
        ArrivalEnum.DM1, ArrivalEnum.DWeibull1, ArrivalEnum.DGamma1,
        ArrivalEnum.MMOODisc
    ]

    for PROCESS in ARRIVAL_PROCESSES:
        print(
            csv_fat_cross_param_power(name="simple_setting",
                                      arrival_enum=PROCESS,
                                      number_flows=2,
                                      number_servers=2,
                                      perform_param=COMMON_PERFORM_PARAM,
                                      opt_method=COMMON_OPTIMIZATION,
                                      mc_dist=MC_EXP1,
                                      compare_metric=COMMON_METRIC,
                                      total_iterations=10**5,
                                      target_util=TARGET_UTIL))

        print(
            csv_fat_cross_param_power(name="simple_setting",
                                      arrival_enum=PROCESS,
                                      number_flows=2,
                                      number_servers=2,
                                      perform_param=COMMON_PERFORM_PARAM,
                                      opt_method=COMMON_OPTIMIZATION,
                                      mc_dist=MC_UNIF10,
                                      compare_metric=COMMON_METRIC,
                                      total_iterations=10**5,
                                      target_util=TARGET_UTIL))
