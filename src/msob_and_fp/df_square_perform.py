"""Compute performance bounds and creates a data frames."""

from typing import List

import pandas as pd
from bound_evaluation.data_frame_to_csv import perform_param_list_to_csv
from nc_arrivals.arrival_distribution import ArrivalDistribution
# from nc_arrivals.qt import DM1
from nc_arrivals.markov_modulated import MMOOCont
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.optimize import Optimize
from utils.perform_param_list import PerformParamList

from msob_and_fp.optimize_fp_bound import OptimizeFPBound
from msob_and_fp.optimize_server_bound import OptimizeServerBound
from msob_and_fp.square import Square


def square_df(arr_list: List[ArrivalDistribution],
              ser_list: List[ConstantRateServer],
              perform_param_list: PerformParamList) -> pd.DataFrame:
    """Compute delay standard_bound for T in T_list and write into dataframe.

    Args:
        arr_list: Arrival object list
        ser_list: Service object list
        perform_param_list: list of performance parameter values

    Returns:
        dataframe

    """
    delta_val = 0.05

    one_param_bounds = [(0.1, 10.0)]
    two_param_bounds = [(0.1, 10.0), (1.1, 10.0)]

    standard_bound = [0.0] * len(perform_param_list)
    server_bound = [0.0] * len(perform_param_list)
    fp_bound = [0.0] * len(perform_param_list)

    for i in range(len(perform_param_list)):
        square_setting = Square(
            arr_list=arr_list,
            ser_list=ser_list,
            perform_param=perform_param_list.get_parameter_at_i(i))

        standard_bound[i] = Optimize(setting=square_setting,
                                     number_param=2).grid_search(
                                         grid_bounds=two_param_bounds,
                                         delta=delta_val).obj_value
        server_bound[i] = OptimizeServerBound(setting_msob_fp=square_setting,
                                              number_param=1).grid_search(
                                                  grid_bounds=one_param_bounds,
                                                  delta=delta_val).obj_value
        fp_bound[i] = OptimizeFPBound(setting_msob_fp=square_setting,
                                      number_param=2).grid_search(
                                          grid_bounds=two_param_bounds,
                                          delta=delta_val).obj_value

    results_df = pd.DataFrame(
        {
            "standard_bound": standard_bound,
            "server_bound": server_bound,
            "fp_bound": fp_bound
        },
        index=perform_param_list.values_list)

    print(f"utilization: {square_setting.approximate_utilization()}")

    return results_df


if __name__ == '__main__':
    # DELAY_PROB_LIST = PerformParamList(perform_metric=PerformEnum.DELAY_PROB,
    #                                    values_list=range(4, 11))
    DELAY_LIST = PerformParamList(perform_metric=PerformEnum.DELAY,
                                  values_list=[10**(-x) for x in range(10)])

    # print(
    #     perform_param_list_to_csv(prefix="square_",
    #                               data_frame_creator=square_df,
    #                               arr_list=[
    #                                   DM1(lamb=0.3),
    #                                   DM1(lamb=6.5),
    #                                   DM1(lamb=5.1),
    #                                   DM1(lamb=1.4)
    #                               ],
    #                               ser_list=[
    #                                   ConstantRateServer(rate=7.0),
    #                                   ConstantRateServer(rate=7.7),
    #                                   ConstantRateServer(rate=5.1),
    #                                   ConstantRateServer(rate=3.1)
    #                               ],
    #                               perform_param_list=DELAY_LIST))
    #
    # print(
    #     perform_param_list_to_csv(prefix="square_",
    #                               data_frame_creator=square_df,
    #                               arr_list=[
    #                                   DM1(lamb=5.1),
    #                                   DM1(lamb=8.3),
    #                                   DM1(lamb=0.2),
    #                                   DM1(lamb=3.6)
    #                               ],
    #                               ser_list=[
    #                                   ConstantRateServer(rate=9.8),
    #                                   ConstantRateServer(rate=8.8),
    #                                   ConstantRateServer(rate=9.5),
    #                                   ConstantRateServer(rate=8.6)
    #                               ],
    #                               perform_param_list=DELAY_LIST))

    # print(
    #     perform_param_list_to_csv(prefix="square_",
    #                               data_frame_creator=square_df,
    #                               arr_list=[
    #                                   MD1(lamb=0.4, mu=1.0),
    #                                   MD1(lamb=3.0, mu=1.0),
    #                                   MD1(lamb=0.4, mu=1.0),
    #                                   MD1(lamb=0.7, mu=1.0)
    #                               ],
    #                               ser_list=[
    #                                   ConstantRateServer(rate=4.8),
    #                                   ConstantRateServer(rate=8.0),
    #                                   ConstantRateServer(rate=6.0),
    #                                   ConstantRateServer(rate=4.5)
    #                               ],
    #                               perform_param_list=DELAY_LIST))

    # print(
    #     perform_param_list_to_csv(prefix="square_",
    #                               data_frame_creator=square_df,
    #                               arr_list=[
    #                                   MMOOFluid(mu=6.4,
    #                                             lamb=3.7,
    #                                             peak_rate=5.8),
    #                                   MMOOFluid(mu=1.8,
    #                                             lamb=7.3,
    #                                             peak_rate=9.9),
    #                                   MMOOFluid(mu=3.6,
    #                                             lamb=6.9,
    #                                             peak_rate=2.3),
    #                                   MMOOFluid(mu=0.4,
    #                                             lamb=1.3,
    #                                             peak_rate=0.7)
    #                               ],
    #                               ser_list=[
    #                                   ConstantRateServer(rate=5.0),
    #                                   ConstantRateServer(rate=6.5),
    #                                   ConstantRateServer(rate=9.1),
    #                                   ConstantRateServer(rate=9.9)
    #                               ],
    #                               perform_param_list=DELAY_LIST))

    print(
        perform_param_list_to_csv(prefix="square_",
                                  data_frame_creator=square_df,
                                  arr_list=[
                                      MMOOCont(mu=7.2, lamb=4.8,
                                               peak_rate=4.9),
                                      MMOOCont(mu=2.5, lamb=2.1,
                                               peak_rate=3.7),
                                      MMOOCont(mu=4.1, lamb=3.1,
                                               peak_rate=1.3),
                                      MMOOCont(mu=3.3, lamb=3.3, peak_rate=1.7)
                                  ],
                                  ser_list=[
                                      ConstantRateServer(rate=5.2),
                                      ConstantRateServer(rate=6.5),
                                      ConstantRateServer(rate=9.9),
                                      ConstantRateServer(rate=3.0)
                                  ],
                                  perform_param_list=DELAY_LIST))
