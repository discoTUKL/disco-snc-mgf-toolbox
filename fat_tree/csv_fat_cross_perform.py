"""Compute delay bound and write into csv file."""

import csv
from typing import List

import pandas as pd

from fat_tree.fat_cross_perform import FatCrossPerform
from library.perform_param_list import PerformParamList
from nc_operations.perform_metric import PerformMetric
from nc_processes.arrival_distribution import (MMOO, ArrivalDistribution,
                                               ExponentialArrival)
from nc_processes.service_distribution import ConstantRate, ServiceDistribution
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize
from optimization.optimize_new import OptimizeNew


def fat_cross_df(arr_list: List[ArrivalDistribution],
                 ser_list: List[ServiceDistribution], opt_method: OptMethod,
                 perform_param_list: PerformParamList) -> pd.DataFrame:
    """Compute delay bound for T in T_list and write into dataframe.

    Args:
        arr_list: Arrival object list
        ser_list: Service object list
        opt_method: method name as string, PS or GS
        perform_param_list: list of performance parameter values

    Returns:
        dataframe

    """
    bound = [0.0] * len(perform_param_list.values_list)
    new_bound = [0.0] * len(perform_param_list.values_list)

    for _i in range(len(perform_param_list.values_list)):
        perform_param = perform_param_list.get_parameter_at_i(_i)
        setting = FatCrossPerform(
            arr_list=arr_list, ser_list=ser_list, perform_param=perform_param)

        if opt_method == OptMethod.GRID_SEARCH:
            bound[_i] = Optimize(setting=setting).grid_search(
                bound_list=[(0.1, 5.0)], delta=0.1)
            new_bound[_i] = OptimizeNew(
                setting_new=setting, new=True).grid_search(
                    bound_list=[(0.1, 5.0), (0.9, 6.0)], delta=0.1)

        elif opt_method == OptMethod.PATTERN_SEARCH:
            bound[_i] = Optimize(setting=setting).pattern_search(
                start_list=[0.5], delta=3.0, delta_min=0.01)

            new_bound[_i] = OptimizeNew(
                setting_new=setting, new=True).pattern_search(
                    start_list=[0.5, 2.0], delta=3.0, delta_min=0.01)

        elif opt_method == OptMethod.GS_OLD:
            bound[_i] = Optimize(setting=setting).grid_search_old(
                bound_list=[(0.1, 5.0)], delta=0.1)
            new_bound[_i] = OptimizeNew(
                setting_new=setting, new=True).grid_search_old(
                    bound_list=[(0.1, 5.0), (0.9, 6.0)], delta=0.1)

        else:
            raise ValueError(
                "Optimization parameter {0} is infeasible".format(opt_method))

    results_df = pd.DataFrame(
        {
            "bound": bound,
            "new_bound": new_bound
        },
        index=perform_param_list.values_list)
    results_df = results_df[["bound", "new_bound"]]

    return results_df


def csv_fat_cross_perform(
        foi_arrival: ArrivalDistribution, cross_arrival: ArrivalDistribution,
        foi_service: ServiceDistribution, cross_service: ServiceDistribution,
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
        filename = "simple_setting_{0}".format(
            perform_param_list.perform_metric.name)
    else:
        filename = "fat_cross_{0}".format(
            perform_param_list.perform_metric.name)

    arr_list: List[ArrivalDistribution] = [foi_arrival]
    ser_list: List[ServiceDistribution] = [foi_service]

    for _i in range(number_servers - 1):
        arr_list.append(cross_arrival)
        ser_list.append(cross_service)

    data_frame = fat_cross_df(
        arr_list=arr_list,
        ser_list=ser_list,
        opt_method=opt_method,
        perform_param_list=perform_param_list)

    filename += "_" + foi_arrival.to_string() + "_" + foi_service.to_string(
    ) + "_" + cross_arrival.to_string() + "_" + cross_service.to_string()

    data_frame.to_csv(
        filename + '.csv', index=True, quoting=csv.QUOTE_NONNUMERIC)

    return data_frame


if __name__ == '__main__':
    DELAY_PROB_LIST = PerformParamList(
        perform_metric=PerformMetric.DELAY_PROB, values_list=range(4, 11))

    exp1 = ExponentialArrival(lamb=0.2)
    exp2 = ExponentialArrival(lamb=8.0)
    # const_rate1 = ConstantRate(rate=8.0)
    # const_rate2 = ConstantRate(rate=0.2)
    #
    # print(
    #     csv_tandem_perform(
    #         foi_arrival=exp1,
    #         cross_arrival=exp2,
    #         foi_service=const_rate1,
    #         cross_service=const_rate2,
    #         number_servers=2,
    #         perform_param_list=DELAY_PROB_LIST,
    #         opt_method=OptMethod.GRID_SEARCH))

    # exp1 = ExponentialArrival(lamb=0.4)
    # exp2 = ExponentialArrival(lamb=3.5)
    # const_rate1 = ConstantRate(rate=4.5)
    # const_rate2 = ConstantRate(rate=0.4)
    #
    # print(
    #     csv_tandem_perform(
    #         foi_arrival=exp1,
    #         cross_arrival=exp2,
    #         foi_service=const_rate1,
    #         cross_service=const_rate2,
    #         number_servers=2,
    #         perform_param_list=DELAY_PROB_LIST,
    #         opt_method=OptMethod.GRID_SEARCH))

    # mmoo_foi = MMOO(mu=1.2, lamb=2.1, burst=3.5)
    # mmoo_2 = MMOO(mu=3.7, lamb=1.5, burst=0.4)
    # foi_rate1 = ConstantRate(rate=2.0)
    # const_rate2 = ConstantRate(rate=0.3)

    # print(
    #     csv_tandem_perform(
    #         foi_arrival=mmoo_foi,
    #         cross_arrival=mmoo_2,
    #         foi_service=foi_rate1,
    #         cross_service=const_rate2,
    #         number_servers=2,
    #         perform_param_list=DELAY_PROB_LIST,
    #         opt_method=OptMethod.GRID_SEARCH))

    mmoo_foi = MMOO(mu=1.0, lamb=2.2, burst=3.4)
    mmoo_2 = MMOO(mu=3.6, lamb=1.6, burst=0.4)
    foi_rate1 = ConstantRate(rate=2.0)
    const_rate2 = ConstantRate(rate=0.3)

    print(
        csv_fat_cross_perform(
            foi_arrival=mmoo_foi,
            cross_arrival=mmoo_2,
            foi_service=foi_rate1,
            cross_service=const_rate2,
            number_servers=2,
            perform_param_list=DELAY_PROB_LIST,
            opt_method=OptMethod.GRID_SEARCH))
