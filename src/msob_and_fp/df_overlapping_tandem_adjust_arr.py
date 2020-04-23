"""Compute performance bounds and creates a data frames."""

from typing import List

import pandas as pd

from bound_evaluation.data_frame_to_csv import arrival_list_to_csv
from msob_and_fp.optimize_fp_bound import OptimizeFPBound
from msob_and_fp.optimize_server_bound import OptimizeServerBound
from msob_and_fp.overlapping_tandem_perform import OverlappingTandemPerform
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.markov_modulated import MMOOFluid
# from nc_arrivals.qt import DM1
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.optimize import Optimize
from utils.perform_parameter import PerformParameter


def overlapping_tandem_adjust_arr_df(
        list_arr_list: List[List[ArrivalDistribution]],
        ser_list: List[ConstantRateServer], server_index: int,
        perform_param: PerformParameter) -> pd.DataFrame:
    """Compute delay standard_bound for T in T_list and write into dataframe.

    Args:
        list_arr_list: different Arrival object lists
        ser_list: Service object list
        server_index: index of the server to be analyzed
        perform_param: performance parameter

    Returns:
        dataframe

    """
    delta_val = 0.05

    one_param_bounds = [(0.1, 10.0)]
    two_param_bounds = [(0.1, 10.0), (1.1, 10.0)]

    standard_bound = [0.0] * len(list_arr_list)
    server_bound = [0.0] * len(list_arr_list)
    fp_bound = [0.0] * len(list_arr_list)
    utilizations = [0.0] * len(list_arr_list)

    for i in range(len(list_arr_list)):
        overlapping_tandem_setting = OverlappingTandemPerform(
            arr_list=list_arr_list[i],
            ser_list=ser_list,
            perform_param=perform_param)

        standard_bound[i] = Optimize(setting=overlapping_tandem_setting,
                                     number_param=2).grid_search(
                                         bound_list=two_param_bounds,
                                         delta=delta_val)
        server_bound[i] = OptimizeServerBound(
            setting_msob_fp=overlapping_tandem_setting,
            number_param=1).grid_search(bound_list=one_param_bounds,
                                        delta=delta_val)
        fp_bound[i] = OptimizeFPBound(
            setting_msob_fp=overlapping_tandem_setting,
            number_param=1).grid_search(bound_list=one_param_bounds,
                                        delta=delta_val)

        utilizations[i] = overlapping_tandem_setting.server_util(
            server_index=server_index)

    results_df = pd.DataFrame(
        {
            "standard_bound": standard_bound,
            "server_bound": server_bound,
            "fp_bound": fp_bound
        },
        index=utilizations)
    results_df = results_df[["standard_bound", "server_bound", "fp_bound"]]

    return results_df


if __name__ == '__main__':
    DELAY3 = PerformParameter(perform_metric=PerformEnum.DELAY, value=10**(-3))

    # print(
    #     arrival_list_to_csv(
    #         prefix="overlapping_tandem_",
    #         data_frame_creator=overlapping_tandem_adjust_arr_df,
    #         list_arr_list=[[DM1(lamb=2.3),
    #                         DM1(lamb=4.5),
    #                         DM1(lamb=0.8)],
    #                        [DM1(lamb=2.3),
    #                         DM1(lamb=4.5),
    #                         DM1(lamb=0.7)],
    #                        [DM1(lamb=2.3),
    #                         DM1(lamb=4.5),
    #                         DM1(lamb=0.6)],
    #                        [DM1(lamb=2.3),
    #                         DM1(lamb=4.5),
    #                         DM1(lamb=0.5)],
    #                        [DM1(lamb=2.3),
    #                         DM1(lamb=4.5),
    #                         DM1(lamb=0.45)],
    #                        [DM1(lamb=2.3),
    #                         DM1(lamb=4.5),
    #                         DM1(lamb=0.4)],
    #                        [DM1(lamb=2.3),
    #                         DM1(lamb=4.5),
    #                         DM1(lamb=0.35)],
    #                        [DM1(lamb=2.3),
    #                         DM1(lamb=4.5),
    #                         DM1(lamb=0.33)],
    #                        [DM1(lamb=2.3),
    #                         DM1(lamb=4.5),
    #                         DM1(lamb=0.31)],
    #                        [DM1(lamb=2.3),
    #                         DM1(lamb=4.5),
    #                         DM1(lamb=0.3)],
    #                        [DM1(lamb=2.3),
    #                         DM1(lamb=4.5),
    #                         DM1(lamb=0.29)],
    #                        [DM1(lamb=2.3),
    #                         DM1(lamb=4.5),
    #                         DM1(lamb=0.28)]],
    #         ser_list=[
    #             ConstantRateServer(rate=1.2),
    #             ConstantRateServer(rate=6.2),
    #             ConstantRateServer(rate=7.3)
    #         ],
    #         server_index=1,
    #         perform_param=DELAY3))

    print(
        arrival_list_to_csv(
            prefix="overlapping_tandem_",
            data_frame_creator=overlapping_tandem_adjust_arr_df,
            list_arr_list=[[
                MMOOFluid(mu=0.8, lamb=7.0, peak_rate=2.0),
                MMOOFluid(mu=4.0, lamb=8.5, peak_rate=5.3),
                MMOOFluid(mu=0.5, lamb=6.0, peak_rate=9.0)
            ],
                           [
                               MMOOFluid(mu=0.8, lamb=7.0, peak_rate=2.0),
                               MMOOFluid(mu=4.0, lamb=8.5, peak_rate=5.3),
                               MMOOFluid(mu=0.5, lamb=5.0, peak_rate=9.0)
                           ],
                           [
                               MMOOFluid(mu=0.8, lamb=7.0, peak_rate=2.0),
                               MMOOFluid(mu=4.0, lamb=8.5, peak_rate=5.3),
                               MMOOFluid(mu=0.5, lamb=4.0, peak_rate=9.0)
                           ],
                           [
                               MMOOFluid(mu=0.8, lamb=7.0, peak_rate=2.0),
                               MMOOFluid(mu=4.0, lamb=8.5, peak_rate=5.3),
                               MMOOFluid(mu=0.5, lamb=3.0, peak_rate=9.0)
                           ],
                           [
                               MMOOFluid(mu=0.8, lamb=7.0, peak_rate=2.0),
                               MMOOFluid(mu=4.0, lamb=8.5, peak_rate=5.3),
                               MMOOFluid(mu=0.5, lamb=2.5, peak_rate=9.0)
                           ],
                           [
                               MMOOFluid(mu=0.8, lamb=7.0, peak_rate=2.0),
                               MMOOFluid(mu=4.0, lamb=8.5, peak_rate=5.3),
                               MMOOFluid(mu=0.5, lamb=2.0, peak_rate=9.0)
                           ],
                           [
                               MMOOFluid(mu=0.8, lamb=7.0, peak_rate=2.0),
                               MMOOFluid(mu=4.0, lamb=8.5, peak_rate=5.3),
                               MMOOFluid(mu=0.5, lamb=1.8, peak_rate=9.0)
                           ],
                           [
                               MMOOFluid(mu=0.8, lamb=7.0, peak_rate=2.0),
                               MMOOFluid(mu=4.0, lamb=8.5, peak_rate=5.3),
                               MMOOFluid(mu=0.5, lamb=1.5, peak_rate=9.0)
                           ],
                           [
                               MMOOFluid(mu=0.8, lamb=7.0, peak_rate=2.0),
                               MMOOFluid(mu=4.0, lamb=8.5, peak_rate=5.3),
                               MMOOFluid(mu=0.5, lamb=1.3, peak_rate=9.0)
                           ],
                           [
                               MMOOFluid(mu=0.8, lamb=7.0, peak_rate=2.0),
                               MMOOFluid(mu=4.0, lamb=8.5, peak_rate=5.3),
                               MMOOFluid(mu=0.5, lamb=1.2, peak_rate=9.0)
                           ],
                           [
                               MMOOFluid(mu=0.8, lamb=7.0, peak_rate=2.0),
                               MMOOFluid(mu=4.0, lamb=8.5, peak_rate=5.3),
                               MMOOFluid(mu=0.5, lamb=1.1, peak_rate=9.0)
                           ],
                           [
                               MMOOFluid(mu=0.8, lamb=7.0, peak_rate=2.0),
                               MMOOFluid(mu=4.0, lamb=8.5, peak_rate=5.3),
                               MMOOFluid(mu=0.5, lamb=1.0, peak_rate=9.0)
                           ],
                           [
                               MMOOFluid(mu=0.8, lamb=7.0, peak_rate=2.0),
                               MMOOFluid(mu=4.0, lamb=8.5, peak_rate=5.3),
                               MMOOFluid(mu=0.5, lamb=0.95, peak_rate=9.0)
                           ],
                           [
                               MMOOFluid(mu=0.8, lamb=7.0, peak_rate=2.0),
                               MMOOFluid(mu=4.0, lamb=8.5, peak_rate=5.3),
                               MMOOFluid(mu=0.5, lamb=0.9, peak_rate=9.0)
                           ]],
            ser_list=[
                ConstantRateServer(rate=2.2),
                ConstantRateServer(rate=7.0),
                ConstantRateServer(rate=10.0)
            ],
            server_index=1,
            perform_param=DELAY3))
