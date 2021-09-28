"""Compute performance bounds and creates a data frames."""

from typing import List

import pandas as pd

from bound_evaluation.data_frame_to_csv import perform_param_list_to_csv
from msob_and_fp.optimize_fp_bound import OptimizeFPBound
from msob_and_fp.optimize_server_bound import OptimizeServerBound
from msob_and_fp.overlapping_tandem_perform import OverlappingTandemPerform
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.iid import DM1
from nc_arrivals.markov_modulated import MMOODisc, MMOOCont
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.optimize import Optimize
from utils.perform_param_list import PerformParamList


def overlapping_tandem_df(
        arr_list: List[ArrivalDistribution],
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
        overlapping_tandem_setting = OverlappingTandemPerform(
            arr_list=arr_list,
            ser_list=ser_list,
            perform_param=perform_param_list.get_parameter_at_i(i))

        standard_bound[i] = Optimize(setting=overlapping_tandem_setting,
                                     number_param=2).grid_search(
                                         grid_bounds=two_param_bounds,
                                         delta=delta_val)
        server_bound[i] = OptimizeServerBound(
            setting_msob_fp=overlapping_tandem_setting,
            number_param=1).grid_search(grid_bounds=one_param_bounds,
                                        delta=delta_val)
        fp_bound[i] = OptimizeFPBound(
            setting_msob_fp=overlapping_tandem_setting,
            number_param=1).grid_search(grid_bounds=one_param_bounds,
                                        delta=delta_val)

    results_df = pd.DataFrame(
        {
            "standard_bound": standard_bound,
            "server_bound": server_bound,
            "fp_bound": fp_bound,
        },
        index=perform_param_list.values_list)
    results_df = results_df[["standard_bound", "server_bound", "fp_bound"]]

    print(
        f"utilization: {overlapping_tandem_setting.approximate_utilization()}")

    return results_df


if __name__ == '__main__':
    # DELAY_PROB_LIST = PerformParamList(perform_metric=PerformEnum.DELAY_PROB,
    #                                    values_list=range(4, 11))
    DELAY_LIST = PerformParamList(perform_metric=PerformEnum.DELAY,
                                  values_list=[10**(-x) for x in range(10)])

    print(
        perform_param_list_to_csv(
            prefix="overlapping_tandem_",
            data_frame_creator=overlapping_tandem_df,
            arr_list=[DM1(lamb=2.0),
                      DM1(lamb=1.0),
                      DM1(lamb=2.0)],
            ser_list=[
                ConstantRateServer(rate=3.5),
                ConstantRateServer(rate=5.0),
                ConstantRateServer(rate=6.0)
            ],
            perform_param_list=DELAY_LIST))

    print(
        perform_param_list_to_csv(
            prefix="overlapping_tandem_",
            data_frame_creator=overlapping_tandem_df,
            arr_list=[DM1(lamb=2.3),
                      DM1(lamb=4.5),
                      DM1(lamb=1.7)],
            ser_list=[
                ConstantRateServer(rate=1.2),
                ConstantRateServer(rate=6.2),
                ConstantRateServer(rate=7.3)
            ],
            perform_param_list=DELAY_LIST))

    print(
        perform_param_list_to_csv(
            prefix="overlapping_tandem_",
            data_frame_creator=overlapping_tandem_df,
            arr_list=[DM1(lamb=2.0),
                      DM1(lamb=2.4),
                      DM1(lamb=5.3)],
            ser_list=[
                ConstantRateServer(rate=4.3),
                ConstantRateServer(rate=2.1),
                ConstantRateServer(rate=5.8)
            ],
            perform_param_list=DELAY_LIST))

    print(
        perform_param_list_to_csv(prefix="overlapping_tandem_",
                                  data_frame_creator=overlapping_tandem_df,
                                  arr_list=[
                                      MMOOCont(mu=7.3, lamb=3.4, peak_rate=1.2),
                                      MMOOCont(mu=1.5, lamb=3.6, peak_rate=8.2),
                                      MMOOCont(mu=1.3, lamb=1.1, peak_rate=0.4)
                                  ],
                                  ser_list=[
                                      ConstantRateServer(rate=9.0),
                                      ConstantRateServer(rate=4.1),
                                      ConstantRateServer(rate=4.7)
                                  ],
                                  perform_param_list=DELAY_LIST))

    print(
        perform_param_list_to_csv(prefix="overlapping_tandem_",
                                  data_frame_creator=overlapping_tandem_df,
                                  arr_list=[
                                      MMOOCont(mu=2.0,
                                               lamb=7.0,
                                               peak_rate=5.0),
                                      MMOOCont(mu=4.0,
                                               lamb=8.5,
                                               peak_rate=5.3),
                                      MMOOCont(mu=0.5,
                                               lamb=4.0,
                                               peak_rate=9.0)
                                  ],
                                  ser_list=[
                                      ConstantRateServer(rate=4.0),
                                      ConstantRateServer(rate=7.0),
                                      ConstantRateServer(rate=10.0)
                                  ],
                                  perform_param_list=DELAY_LIST))

    print(
        perform_param_list_to_csv(prefix="overlapping_tandem_",
                                  data_frame_creator=overlapping_tandem_df,
                                  arr_list=[
                                      MMOODisc(stay_on=0.6,
                                               stay_off=0.4,
                                               peak_rate=0.9),
                                      MMOODisc(stay_on=0.6,
                                               stay_off=0.4,
                                               peak_rate=0.5),
                                      MMOODisc(stay_on=0.6,
                                               stay_off=0.4,
                                               peak_rate=0.7)
                                  ],
                                  ser_list=[
                                      ConstantRateServer(rate=1.2),
                                      ConstantRateServer(rate=6.2),
                                      ConstantRateServer(rate=7.3)
                                  ],
                                  perform_param_list=DELAY_LIST))
