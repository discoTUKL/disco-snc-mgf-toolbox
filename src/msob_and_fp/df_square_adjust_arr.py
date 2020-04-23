"""Compute performance bounds and creates a data frames."""

from typing import List

import pandas as pd

from bound_evaluation.data_frame_to_csv import arrival_list_to_csv
from msob_and_fp.optimize_fp_bound import OptimizeFPBound
from msob_and_fp.optimize_server_bound import OptimizeServerBound
from msob_and_fp.square_perform import SquarePerform
from nc_arrivals.arrival_distribution import ArrivalDistribution
# from nc_arrivals.qt import DM1
from nc_arrivals.markov_modulated import MMOOFluid
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.optimize import Optimize
from utils.perform_parameter import PerformParameter


def square_adjust_arr_df(list_arr_list: List[List[ArrivalDistribution]],
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
        overlapping_tandem_setting = SquarePerform(arr_list=list_arr_list[i],
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
            number_param=2).grid_search(bound_list=two_param_bounds,
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
    # DELAY_PROB15 = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
    #                                 value=15)
    DELAY3 = PerformParameter(perform_metric=PerformEnum.DELAY, value=10**(-3))

    # print(
    #     arrival_list_to_csv(
    #         prefix="square_",
    #         data_frame_creator=square_adjust_arr_df,
    #         list_arr_list=[
    #             [DM1(lamb=5.1),
    #              DM1(lamb=8.3),
    #              DM1(lamb=0.6),
    #              DM1(lamb=3.6)],
    #             [DM1(lamb=5.1),
    #              DM1(lamb=8.3),
    #              DM1(lamb=0.5),
    #              DM1(lamb=3.6)],
    #             [DM1(lamb=5.1),
    #              DM1(lamb=8.3),
    #              DM1(lamb=0.4),
    #              DM1(lamb=3.6)],
    #             [DM1(lamb=5.1),
    #              DM1(lamb=8.3),
    #              DM1(lamb=0.3),
    #              DM1(lamb=3.6)],
    #             [DM1(lamb=5.1),
    #              DM1(lamb=8.3),
    #              DM1(lamb=0.25),
    #              DM1(lamb=3.6)],
    #             [DM1(lamb=5.1),
    #              DM1(lamb=8.3),
    #              DM1(lamb=0.2),
    #              DM1(lamb=3.6)],
    #             [DM1(lamb=5.1),
    #              DM1(lamb=8.3),
    #              DM1(lamb=0.15),
    #              DM1(lamb=3.6)],
    #             [DM1(lamb=5.1),
    #              DM1(lamb=8.3),
    #              DM1(lamb=0.13),
    #              DM1(lamb=3.6)],
    #             [DM1(lamb=5.1),
    #              DM1(lamb=8.3),
    #              DM1(lamb=0.12),
    #              DM1(lamb=3.6)]
    #         ],
    #         ser_list=[
    #             ConstantRateServer(rate=9.8),
    #             ConstantRateServer(rate=8.8),
    #             ConstantRateServer(rate=9.5),
    #             ConstantRateServer(rate=8.6)
    #         ],
    #         perform_param=DELAY3))

    # print(
    #     arrival_list_to_csv(
    #         prefix="square_",
    #         data_frame_creator=square_adjust_arr_df,
    #         list_arr_list=[
    #             [MMOOFluid(mu=7.0, lamb=4.8, burst=4.9),
    #              MMOOFluid(mu=2.2, lamb=2.1, burst=3.7),
    #              MMOOFluid(mu=4.1, lamb=1.0, burst=0.5),
    #              MMOOFluid(mu=2.5, lamb=3.3, burst=1.6)],
    #             [MMOOFluid(mu=7.0, lamb=4.8, burst=4.9),
    #              MMOOFluid(mu=2.2, lamb=2.1, burst=3.7),
    #              MMOOFluid(mu=4.1, lamb=1.0, burst=0.8),
    #              MMOOFluid(mu=2.5, lamb=3.3, burst=1.6)],
    #             [MMOOFluid(mu=7.0, lamb=4.8, burst=4.9),
    #              MMOOFluid(mu=2.2, lamb=2.1, burst=3.7),
    #              MMOOFluid(mu=4.1, lamb=1.0, burst=1.0),
    #              MMOOFluid(mu=2.5, lamb=3.3, burst=1.6)],
    #             [MMOOFluid(mu=7.0, lamb=4.8, burst=4.9),
    #              MMOOFluid(mu=2.2, lamb=2.1, burst=3.7),
    #              MMOOFluid(mu=4.1, lamb=1.0, burst=1.5),
    #              MMOOFluid(mu=2.5, lamb=3.3, burst=1.6)],
    #             [MMOOFluid(mu=7.0, lamb=4.8, burst=4.9),
    #              MMOOFluid(mu=2.2, lamb=2.1, burst=3.7),
    #              MMOOFluid(mu=4.1, lamb=1.0, burst=1.8),
    #              MMOOFluid(mu=2.5, lamb=3.3, burst=1.6)],
    #             [MMOOFluid(mu=7.0, lamb=4.8, burst=4.9),
    #              MMOOFluid(mu=2.2, lamb=2.1, burst=3.7),
    #              MMOOFluid(mu=4.1, lamb=1.0, burst=2.0),
    #              MMOOFluid(mu=2.5, lamb=3.3, burst=1.6)],
    #             [MMOOFluid(mu=7.0, lamb=4.8, burst=4.9),
    #              MMOOFluid(mu=2.2, lamb=2.1, burst=3.7),
    #              MMOOFluid(mu=4.1, lamb=1.0, burst=2.2),
    #              MMOOFluid(mu=2.5, lamb=3.3, burst=1.6)],
    #             [MMOOFluid(mu=7.0, lamb=4.8, burst=4.9),
    #              MMOOFluid(mu=2.2, lamb=2.1, burst=3.7),
    #              MMOOFluid(mu=4.1, lamb=1.0, burst=2.3),
    #              MMOOFluid(mu=2.5, lamb=3.3, burst=1.6)],
    #             [MMOOFluid(mu=7.0, lamb=4.8, burst=4.9),
    #              MMOOFluid(mu=2.2, lamb=2.1, burst=3.7),
    #              MMOOFluid(mu=4.1, lamb=1.0, burst=2.5),
    #              MMOOFluid(mu=2.5, lamb=3.3, burst=1.6)]
    #
    #         ],
    #         ser_list=[
    #             ConstantRateServer(rate=5.2),
    #             ConstantRateServer(rate=6.5),
    #             ConstantRateServer(rate=9.9),
    #             ConstantRateServer(rate=3.0)
    #         ],
    #         perform_param=DELAY3))

    print(
        arrival_list_to_csv(prefix="square_",
                            data_frame_creator=square_adjust_arr_df,
                            list_arr_list=[[
                                MMOOFluid(mu=7.0, lamb=4.8, peak_rate=4.9),
                                MMOOFluid(mu=2.2, lamb=2.1, peak_rate=3.7),
                                MMOOFluid(mu=4.1, lamb=1.0, peak_rate=2.0),
                                MMOOFluid(mu=2.5, lamb=3.3, peak_rate=0.1)
                            ],
                                           [
                                               MMOOFluid(mu=7.0,
                                                         lamb=4.8,
                                                         peak_rate=4.9),
                                               MMOOFluid(mu=2.2,
                                                         lamb=2.1,
                                                         peak_rate=3.7),
                                               MMOOFluid(mu=4.1,
                                                         lamb=1.0,
                                                         peak_rate=2.0),
                                               MMOOFluid(mu=2.5,
                                                         lamb=3.3,
                                                         peak_rate=0.3)
                                           ],
                                           [
                                               MMOOFluid(mu=7.0,
                                                         lamb=4.8,
                                                         peak_rate=4.9),
                                               MMOOFluid(mu=2.2,
                                                         lamb=2.1,
                                                         peak_rate=3.7),
                                               MMOOFluid(mu=4.1,
                                                         lamb=1.0,
                                                         peak_rate=2.0),
                                               MMOOFluid(mu=2.5,
                                                         lamb=3.3,
                                                         peak_rate=0.5)
                                           ],
                                           [
                                               MMOOFluid(mu=7.0,
                                                         lamb=4.8,
                                                         peak_rate=4.9),
                                               MMOOFluid(mu=2.2,
                                                         lamb=2.1,
                                                         peak_rate=3.7),
                                               MMOOFluid(mu=4.1,
                                                         lamb=1.0,
                                                         peak_rate=2.0),
                                               MMOOFluid(mu=2.5,
                                                         lamb=3.3,
                                                         peak_rate=1.0)
                                           ],
                                           [
                                               MMOOFluid(mu=7.0,
                                                         lamb=4.8,
                                                         peak_rate=4.9),
                                               MMOOFluid(mu=2.2,
                                                         lamb=2.1,
                                                         peak_rate=3.7),
                                               MMOOFluid(mu=4.1,
                                                         lamb=1.0,
                                                         peak_rate=2.0),
                                               MMOOFluid(mu=2.5,
                                                         lamb=3.3,
                                                         peak_rate=1.3)
                                           ],
                                           [
                                               MMOOFluid(mu=7.0,
                                                         lamb=4.8,
                                                         peak_rate=4.9),
                                               MMOOFluid(mu=2.2,
                                                         lamb=2.1,
                                                         peak_rate=3.7),
                                               MMOOFluid(mu=4.1,
                                                         lamb=1.0,
                                                         peak_rate=2.0),
                                               MMOOFluid(mu=2.5,
                                                         lamb=3.3,
                                                         peak_rate=1.5)
                                           ],
                                           [
                                               MMOOFluid(mu=7.0,
                                                         lamb=4.8,
                                                         peak_rate=4.9),
                                               MMOOFluid(mu=2.2,
                                                         lamb=2.1,
                                                         peak_rate=3.7),
                                               MMOOFluid(mu=4.1,
                                                         lamb=1.0,
                                                         peak_rate=2.0),
                                               MMOOFluid(mu=2.5,
                                                         lamb=3.3,
                                                         peak_rate=1.8)
                                           ],
                                           [
                                               MMOOFluid(mu=7.0,
                                                         lamb=4.8,
                                                         peak_rate=4.9),
                                               MMOOFluid(mu=2.2,
                                                         lamb=2.1,
                                                         peak_rate=3.7),
                                               MMOOFluid(mu=4.1,
                                                         lamb=1.0,
                                                         peak_rate=2.0),
                                               MMOOFluid(mu=2.5,
                                                         lamb=3.3,
                                                         peak_rate=2.0)
                                           ],
                                           [
                                               MMOOFluid(mu=7.0,
                                                         lamb=4.8,
                                                         peak_rate=4.9),
                                               MMOOFluid(mu=2.2,
                                                         lamb=2.1,
                                                         peak_rate=3.7),
                                               MMOOFluid(mu=4.1,
                                                         lamb=1.0,
                                                         peak_rate=2.0),
                                               MMOOFluid(mu=2.5,
                                                         lamb=3.3,
                                                         peak_rate=2.1)
                                           ],
                                           [
                                               MMOOFluid(mu=7.0,
                                                         lamb=4.8,
                                                         peak_rate=4.9),
                                               MMOOFluid(mu=2.2,
                                                         lamb=2.1,
                                                         peak_rate=3.7),
                                               MMOOFluid(mu=4.1,
                                                         lamb=1.0,
                                                         peak_rate=2.0),
                                               MMOOFluid(mu=2.5,
                                                         lamb=3.3,
                                                         peak_rate=2.2)
                                           ],
                                           [
                                               MMOOFluid(mu=7.0,
                                                         lamb=4.8,
                                                         peak_rate=4.9),
                                               MMOOFluid(mu=2.2,
                                                         lamb=2.1,
                                                         peak_rate=3.7),
                                               MMOOFluid(mu=4.1,
                                                         lamb=1.0,
                                                         peak_rate=2.0),
                                               MMOOFluid(mu=2.5,
                                                         lamb=3.3,
                                                         peak_rate=2.3)
                                           ]],
                            ser_list=[
                                ConstantRateServer(rate=5.2),
                                ConstantRateServer(rate=6.5),
                                ConstantRateServer(rate=9.9),
                                ConstantRateServer(rate=3.0)
                            ],
                            server_index=3,
                            perform_param=DELAY3))
