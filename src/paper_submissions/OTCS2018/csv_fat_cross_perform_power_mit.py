"""Compute delay bound and write into csv file."""

import csv
from typing import List

import pandas as pd

from fat_tree.fat_cross_perform import FatCrossPerform
from h_mitigator.optimize_mitigator import OptimizeMitigator
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.markov_modulated import MMOOFluid
from nc_arrivals.qt import DM1, MD1
from nc_operations.perform_enum import PerformEnum
from nc_service.constant_rate_server import ConstantRate
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize
from utils.perform_param_list import PerformParamList


def fat_cross_df(arr_list: List[ArrivalDistribution],
                 ser_list: List[ConstantRate], opt_method: OptMethod,
                 perform_param_list: PerformParamList) -> pd.DataFrame:
    """Compute delay bound for T in T_list and write into dataframe.

    Args:
        arr_list: Arrival object list
        ser_list: Service object list
        opt_method: PS or GS
        perform_param_list: list of performance parameter values

    Returns:
        dataframe

    """
    bound = [0.0] * len(perform_param_list)
    new_bound = [0.0] * len(perform_param_list)

    for i in range(len(perform_param_list)):
        perform_param = perform_param_list.get_parameter_at_i(i)
        setting = FatCrossPerform(
            arr_list=arr_list, ser_list=ser_list, perform_param=perform_param)

        if opt_method == OptMethod.GRID_SEARCH:
            bound[i] = Optimize(setting=setting).grid_search(
                bound_list=[(0.1, 5.0)], delta=0.1)
            new_bound[i] = OptimizeMitigator(
                setting_h_mit=setting).grid_search(
                    bound_list=[(0.1, 5.0), (0.9, 6.0)], delta=0.05)

        elif opt_method == OptMethod.PATTERN_SEARCH:
            bound[i] = Optimize(setting=setting).pattern_search(
                start_list=[0.5], delta=3.0, delta_min=0.01)

            new_bound[i] = OptimizeMitigator(
                setting_h_mit=setting).pattern_search(
                    start_list=[0.5, 2.0], delta=3.0, delta_min=0.01)

        elif opt_method == OptMethod.GS_OLD:
            bound[i] = Optimize(setting=setting).grid_search_old(
                bound_list=[(0.1, 5.0)], delta=0.1)
            new_bound[i] = OptimizeMitigator(
                setting_h_mit=setting).grid_search_old(
                    bound_list=[(0.1, 5.0), (0.9, 6.0)], delta=0.1)

        else:
            raise ValueError(
                "Optimization parameter {0} is infeasible".format(opt_method))

    results_df = pd.DataFrame({
        "bound": bound,
        "h_mit_bound": new_bound
    },
                              index=perform_param_list.values_list)
    results_df = results_df[["bound", "h_mit_bound"]]

    return results_df


def csv_fat_cross_perform(
        foi_arrival: ArrivalDistribution, cross_arrival: ArrivalDistribution,
        foi_service: ConstantRate, cross_service: ConstantRate,
        number_servers: int, perform_param_list: PerformParamList,
        opt_method: OptMethod) -> pd.DataFrame:
    """Write dataframe results into a csv file.

    Args:
        foi_arrival: flow of interest's arrival distribution
        cross_arrival: Distribution of cross arrivals
        foi_service: service of the server at the foi
        cross_service: service of remaining servers
        number_servers: number of servers in fat tree
        perform_param_list: list of performance parameter values
        opt_method: optimization method

    Returns:
        csv file

    """
    if number_servers == 2:
        filename = "simple_setting_"
    else:
        filename = "fat_cross_"

    filename += perform_param_list.to_name()

    arr_list: List[ArrivalDistribution] = [foi_arrival]
    ser_list: List[ConstantRate] = [foi_service]

    for _i in range(number_servers - 1):
        arr_list.append(cross_arrival)
        ser_list.append(cross_service)

    data_frame = fat_cross_df(
        arr_list=arr_list,
        ser_list=ser_list,
        opt_method=opt_method,
        perform_param_list=perform_param_list)

    filename += "_" + foi_arrival.to_name() + "_" + foi_arrival.to_value(
        number=1, show_n=False) + "_" + foi_service.to_value(
            number=1) + "_" + cross_arrival.to_value(
                number=2,
                show_n=False) + "_" + cross_service.to_value(number=2)

    data_frame.to_csv(
        filename + '.csv', index=True, quoting=csv.QUOTE_NONNUMERIC)

    return data_frame


if __name__ == '__main__':
    DELAY_PROB_LIST = PerformParamList(
        perform_metric=PerformEnum.DELAY_PROB, values_list=range(4, 11))

    print(
        csv_fat_cross_perform(
            foi_arrival=DM1(lamb=0.2),
            cross_arrival=DM1(lamb=8.0),
            foi_service=ConstantRate(rate=8.0),
            cross_service=ConstantRate(rate=0.2),
            number_servers=2,
            perform_param_list=DELAY_PROB_LIST,
            opt_method=OptMethod.GRID_SEARCH))

    print(
        csv_fat_cross_perform(
            foi_arrival=DM1(lamb=0.4),
            cross_arrival=DM1(lamb=3.5),
            foi_service=ConstantRate(rate=4.5),
            cross_service=ConstantRate(rate=0.4),
            number_servers=2,
            perform_param_list=DELAY_PROB_LIST,
            opt_method=OptMethod.GRID_SEARCH))

    print(
        csv_fat_cross_perform(
            foi_arrival=MMOOFluid(mu=1.2, lamb=2.1, burst=3.5),
            cross_arrival=MMOOFluid(mu=3.7, lamb=1.5, burst=0.4),
            foi_service=ConstantRate(rate=2.0),
            cross_service=ConstantRate(rate=0.3),
            number_servers=2,
            perform_param_list=DELAY_PROB_LIST,
            opt_method=OptMethod.GRID_SEARCH))

    print(
        csv_fat_cross_perform(
            foi_arrival=MMOOFluid(mu=1.0, lamb=2.2, burst=3.4),
            cross_arrival=MMOOFluid(mu=3.6, lamb=1.6, burst=0.4),
            foi_service=ConstantRate(rate=2.0),
            cross_service=ConstantRate(rate=0.3),
            number_servers=2,
            perform_param_list=DELAY_PROB_LIST,
            opt_method=OptMethod.GRID_SEARCH))

    # RATE1 = ConstantRate(rate=1.0)

    print(
        csv_fat_cross_perform(
            foi_arrival=MD1(lamb=1.6, mu=1.0),
            cross_arrival=MD1(lamb=0.01, mu=1.0),
            foi_service=ConstantRate(2.0),
            cross_service=ConstantRate(0.15),
            number_servers=2,
            perform_param_list=DELAY_PROB_LIST,
            opt_method=OptMethod.GRID_SEARCH))

    print(
        csv_fat_cross_perform(
            foi_arrival=MD1(lamb=3.6, mu=1.0),
            cross_arrival=MD1(lamb=0.28, mu=1.0),
            foi_service=ConstantRate(4.4),
            cross_service=ConstantRate(0.7),
            number_servers=2,
            perform_param_list=DELAY_PROB_LIST,
            opt_method=OptMethod.GRID_SEARCH))
