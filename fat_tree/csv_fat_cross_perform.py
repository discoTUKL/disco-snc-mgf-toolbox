"""Compute delay bound for various T and write into csv file."""

import csv
from math import nan
from typing import List

import pandas as pd

from fat_tree.fat_cross_perform import FatCrossPerform
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize
from optimization.optimize_new import OptimizeNew
from library.perform_param_list import PerformParamList
from library.perform_parameter import PerformParameter
from nc_processes.arrival_distribution import (MMOO, ArrivalDistribution,
                                               ExponentialArrival)
from nc_processes.distrib_param import DistribParam
from nc_operations.perform_metric import PerformMetric
from nc_processes.service import ConstantRate, Service


def fat_cross_df(arr_list: List[ArrivalDistribution], ser_list: List[Service],
                 opt_method: OptMethod,
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
    bound = [nan] * len(perform_param_list.values_list)
    new_bound = [nan] * len(perform_param_list.values_list)

    for i, value in enumerate(perform_param_list.values_list):
        perform_param = PerformParameter(
            perform_metric=perform_param_list.perform_metric, value=value)
        setting = FatCrossPerform(
            arr_list=arr_list, ser_list=ser_list, perform_param=perform_param)

        if opt_method == OptMethod.GRID_SEARCH:
            bound[i] = Optimize(setting=setting).grid_search(
                bound_list=[(0.1, 5.0)], delta=0.1)

            new_bound[i] = OptimizeNew(
                setting_new=setting, new=True).grid_search(
                bound_list=[(0.1, 5.0), (0.9, 6.0)], delta=0.1)
        elif opt_method == OptMethod.PATTERN_SEARCH:
            bound[i] = Optimize(setting=setting).pattern_search(
                start_list=[0.5], delta=3.0, delta_min=0.01)

            new_bound[i] = OptimizeNew(
                setting_new=setting, new=True).pattern_search(
                start_list=[0.5, 2.0], delta=3.0, delta_min=0.01)
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


def csv_fat_cross_perform(arrival: ArrivalDistribution,
                          list_of_parameters: List[DistribParam],
                          perform_param_list: PerformParamList,
                          opt_method: OptMethod) -> pd.DataFrame:
    """Write dataframe results into a csv file.

    Args:
        arrival: String that represents the arrival process
        list_of_parameters: dictionaries with actual values
        perform_param_list: list of performance parameter values
        opt_method: optimization method

    Returns:
        csv file

    """
    if len(list_of_parameters) == 2 and isinstance(arrival,
                                                   ExponentialArrival) or len(
            list_of_parameters) == 4 and isinstance(arrival, MMOO):
        filename = "simple_setting_" + perform_param_list.perform_metric.name
    else:
        filename = "fat_cross_" + perform_param_list.perform_metric.name

    if isinstance(arrival, ExponentialArrival):
        lamb1 = list_of_parameters[0].lamb
        lamb2 = list_of_parameters[1].lamb
        rate1 = list_of_parameters[0].rate
        rate2 = list_of_parameters[1].rate

        data_frame = fat_cross_df(
            arr_list=[
                ExponentialArrival(lamb=lamb1),
                ExponentialArrival(lamb=lamb2)
            ],
            ser_list=[ConstantRate(rate=rate1),
                      ConstantRate(rate=rate2)],
            opt_method=opt_method,
            perform_param_list=perform_param_list)

        for item, value in enumerate(list_of_parameters):
            filename += "_" + value.get_exp_string(item)

    elif isinstance(arrival, MMOO):
        mu1 = list_of_parameters[0].mu
        mu2 = list_of_parameters[1].mu
        lamb1 = list_of_parameters[0].lamb
        lamb2 = list_of_parameters[1].lamb
        burst1 = list_of_parameters[0].burst
        burst2 = list_of_parameters[1].burst
        rate1 = list_of_parameters[2].rate
        rate2 = list_of_parameters[3].rate

        data_frame = fat_cross_df(
            arr_list=[
                MMOO(mu=mu1, lamb=lamb1, burst=burst1),
                MMOO(mu=mu2, lamb=lamb2, burst=burst2)
            ],
            ser_list=[ConstantRate(rate=rate1),
                      ConstantRate(rate=rate2)],
            opt_method=OptMethod.GRID_SEARCH,
            perform_param_list=perform_param_list)

        for item, value in enumerate(list_of_parameters):
            filename += "_" + value.get_mmoo_string(item)
    else:
        raise NameError("This arrival process is not implemented")

    data_frame.to_csv(
        filename + '.csv', index=True, quoting=csv.QUOTE_NONNUMERIC)

    return data_frame


if __name__ == '__main__':
    DELAY_PROB_LIST = PerformParamList(
        perform_metric=PerformMetric.DELAY_PROB, values_list=range(4, 11))
    EXP_ARRIVAL1 = ExponentialArrival()
    MMOO1 = MMOO()

    # exp1 = DistribParam(lamb=0.2)
    # exp2 = DistribParam(lamb=8.0)
    # const_rate1 = DistribParam(rate=8.0)
    # const_rate2 = DistribParam(rate=0.2)
    #
    # print(
    #     csv_fat_cross_perform(
    #         arrival=EXP_ARRIVAL1,
    #         list_of_parameters=[exp1, exp2, const_rate1, const_rate2],
    #         perform_param_list=DELAY_PROB_LIST,
    #         opt_method=OptMethod.GRID_SEARCH))

    # exp1 = DistribParam(lamb=0.4)
    # exp2 = DistribParam(lamb=3.5)
    # const_rate1 = DistribParam(rate=4.5)
    # const_rate2 = DistribParam(rate=0.4)
    #
    # print(
    #     csv_fat_cross_perform(
    #         arrival=EXP_ARRIVAL1,
    #         list_of_parameters=[exp1, exp2, const_rate1, const_rate2],
    #         perform_param_list=DELAY_PROB_LIST,
    #         opt_method=OptMethod.GRID_SEARCH))

    # mmoo1 = DistribParam(mu=1.2, lamb=2.1, burst=3.5)
    # mmoo2 = DistribParam(mu=3.7, lamb=1.5, burst=0.4)
    # const_rate1 = DistribParam(rate=2.0)
    # const_rate2 = DistribParam(rate=0.3)
    #
    # print(
    #     csv_fat_cross_perform(
    #         arrival=MMOO1,
    #         list_of_parameters=[mmoo1, mmoo2, const_rate1, const_rate2],
    #         perform_param_list=DELAY_PROB_LIST,
    #         opt_method=OptMethod.GRID_SEARCH))

    mmoo1 = DistribParam(mu=1.0, lamb=2.2, burst=3.4)
    mmoo2 = DistribParam(mu=3.6, lamb=1.6, burst=0.4)
    const_rate1 = DistribParam(rate=2.0)
    const_rate2 = DistribParam(rate=0.3)

    print(
        csv_fat_cross_perform(
            arrival=MMOO1,
            list_of_parameters=[mmoo1, mmoo2, const_rate1, const_rate2],
            perform_param_list=DELAY_PROB_LIST,
            opt_method=OptMethod.GRID_SEARCH))
