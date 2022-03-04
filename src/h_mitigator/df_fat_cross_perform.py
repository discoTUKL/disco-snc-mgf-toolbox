"""Compute delay standard_bound and write into csv file."""

from typing import List

import pandas as pd
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize
from utils.perform_param_list import PerformParamList
from utils.perform_parameter import PerformParameter

from h_mitigator.fat_cross_perform import FatCrossPerform
from h_mitigator.optimize_mitigator import OptimizeMitigator


def fat_cross_perform_df(arr_list: List[ArrivalDistribution],
                         ser_list: List[ConstantRateServer],
                         opt_method: OptMethod,
                         perform_param_list: PerformParamList) -> pd.DataFrame:
    """Compute delay standard_bound for T in T_list and write into dataframe.

    Args:
        arr_list: Arrival object list
        ser_list: Service object list
        opt_method: PS or GS
        perform_param_list: list of performance parameter values

    Returns:
        dataframe

    """
    delta_val = 0.05

    one_param_bounds = [(delta_val, 10.0)]

    standard_bound = [0.0] * len(perform_param_list)
    h_mit_bound = [0.0] * len(perform_param_list)

    fat_cross_setting = FatCrossPerform(
        arr_list=arr_list,
        ser_list=ser_list,
        perform_param=PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                       value=0))

    print(f"utilization: {fat_cross_setting.approximate_utilization()}")
    print()

    for i in range(len(perform_param_list)):
        fat_cross_setting = FatCrossPerform(
            arr_list=arr_list,
            ser_list=ser_list,
            perform_param=perform_param_list.get_parameter_at_i(i))

        if opt_method == OptMethod.GRID_SEARCH:
            standard_bound[i] = Optimize(setting=fat_cross_setting,
                                         number_param=1).grid_search(
                                             grid_bounds=one_param_bounds,
                                             delta=delta_val).obj_value
            h_mit_bound[i] = OptimizeMitigator(
                setting_h_mit=fat_cross_setting,
                number_param=2).grid_search(grid_bounds=[(delta_val, 10.0),
                                                         (1 + delta_val, 8.0)],
                                            delta=delta_val).obj_value

        elif opt_method == OptMethod.PATTERN_SEARCH:
            standard_bound[i] = Optimize(setting=fat_cross_setting,
                                         number_param=1).pattern_search(
                                             start_list=[0.5],
                                             delta=3.0,
                                             delta_min=0.01).obj_value

            h_mit_bound[i] = OptimizeMitigator(setting_h_mit=fat_cross_setting,
                                               number_param=2).pattern_search(
                                                   start_list=[0.5, 2.0],
                                                   delta=3.0,
                                                   delta_min=0.01).obj_value

        else:
            raise NotImplementedError(f"Optimization parameter {opt_method} "
                                      f"is infeasible")

    results_df = pd.DataFrame(
        {
            "standard_bound": standard_bound,
            "h_mit_bound": h_mit_bound
        },
        index=perform_param_list.values_list)

    return results_df


if __name__ == '__main__':
    from bound_evaluation.data_frame_to_csv import perform_param_list_to_csv
    from nc_arrivals.iid import DM1, MD1
    from nc_arrivals.markov_modulated import MMOOCont

    DELAY_PROB_LIST = PerformParamList(perform_metric=PerformEnum.DELAY_PROB,
                                       values_list=list(range(4, 11)))

    print(
        perform_param_list_to_csv(prefix="simple_setting_",
                                  data_frame_creator=fat_cross_perform_df,
                                  arr_list=[DM1(lamb=0.2),
                                            DM1(lamb=8.0)],
                                  ser_list=[
                                      ConstantRateServer(rate=8.0),
                                      ConstantRateServer(rate=0.2)
                                  ],
                                  perform_param_list=DELAY_PROB_LIST,
                                  opt_method=OptMethod.GRID_SEARCH))

    print(
        perform_param_list_to_csv(prefix="simple_setting_",
                                  data_frame_creator=fat_cross_perform_df,
                                  arr_list=[DM1(lamb=0.4),
                                            DM1(lamb=3.5)],
                                  ser_list=[
                                      ConstantRateServer(rate=4.5),
                                      ConstantRateServer(rate=0.4)
                                  ],
                                  perform_param_list=DELAY_PROB_LIST,
                                  opt_method=OptMethod.GRID_SEARCH))

    print(
        perform_param_list_to_csv(prefix="simple_setting_",
                                  data_frame_creator=fat_cross_perform_df,
                                  arr_list=[
                                      MMOOCont(mu=1.2, lamb=2.1,
                                               peak_rate=3.5),
                                      MMOOCont(mu=3.7, lamb=1.5, peak_rate=0.4)
                                  ],
                                  ser_list=[
                                      ConstantRateServer(rate=2.0),
                                      ConstantRateServer(rate=0.3)
                                  ],
                                  perform_param_list=DELAY_PROB_LIST,
                                  opt_method=OptMethod.GRID_SEARCH))

    print(
        perform_param_list_to_csv(prefix="simple_setting_",
                                  data_frame_creator=fat_cross_perform_df,
                                  arr_list=[
                                      MMOOCont(mu=1.0, lamb=2.2,
                                               peak_rate=3.4),
                                      MMOOCont(mu=3.6, lamb=1.6, peak_rate=0.4)
                                  ],
                                  ser_list=[
                                      ConstantRateServer(rate=2.0),
                                      ConstantRateServer(rate=0.3)
                                  ],
                                  perform_param_list=DELAY_PROB_LIST,
                                  opt_method=OptMethod.GRID_SEARCH))

    print(
        perform_param_list_to_csv(
            prefix="simple_setting_",
            data_frame_creator=fat_cross_perform_df,
            arr_list=[MD1(lamb=1.6, mu=1.0),
                      MD1(lamb=0.01, mu=1.0)],
            ser_list=[
                ConstantRateServer(rate=2.0),
                ConstantRateServer(rate=0.15)
            ],
            perform_param_list=DELAY_PROB_LIST,
            opt_method=OptMethod.GRID_SEARCH))

    print(
        perform_param_list_to_csv(
            prefix="simple_setting_",
            data_frame_creator=fat_cross_perform_df,
            arr_list=[MD1(lamb=3.6, mu=1.0),
                      MD1(lamb=0.28, mu=1.0)],
            ser_list=[
                ConstantRateServer(rate=4.4),
                ConstantRateServer(rate=0.7)
            ],
            perform_param_list=DELAY_PROB_LIST,
            opt_method=OptMethod.GRID_SEARCH))
