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
            bound[i] = Optimize(
                setting=setting).grid_search_old(
                    bound_list=[(0.1, 4.0)], delta=0.1)

            new_bound[i] = OptimizeNew(
                setting_new=setting, new=True).grid_search_old(
                    bound_list=[(0.1, 4.0), (0.9, 6.0)], delta=0.1)
        elif opt_method == OptMethod.PATTERN_SEARCH:
            bound[i] = Optimize(
                setting=setting).pattern_search(
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


def csv_fat_cross_perform(arrival: ArrivalDistribution, parameters: dict,
                          perform_param_list: PerformParamList,
                          opt_method: OptMethod) -> pd.DataFrame:
    # TODO: Change parameter dict to list
    """Write dataframe results into a csv file.

    Args:
        arrival: String that represents the arrival process
        parameters: dictionaries with actual values
        perform_param_list: list of performance parameter values
        opt_method: optimization method

    Returns:
        csv file

    """
    filename = "simple_setting_" + perform_param_list.perform_metric.name
    # filename = "fat_cross_" + perform_param_list.perform_metric.name

    if isinstance(arrival, ExponentialArrival):
        lamb1 = parameters["lamb1"]
        lamb2 = parameters["lamb2"]
        rate1 = parameters["rate1"]
        rate2 = parameters["rate2"]

        data_frame = fat_cross_df(
            arr_list=[
                ExponentialArrival(lamb=lamb1),
                ExponentialArrival(lamb=lamb2)
            ],
            ser_list=[ConstantRate(rate=rate1),
                      ConstantRate(rate=rate2)],
            opt_method=opt_method,
            perform_param_list=perform_param_list)
    elif isinstance(arrival, MMOO):
        mu1 = parameters["mu1"]
        mu2 = parameters["mu2"]
        lamb1 = parameters["lamb1"]
        lamb2 = parameters["lamb2"]
        burst1 = parameters["burst1"]
        burst2 = parameters["burst2"]
        rate1 = parameters["rate1"]
        rate2 = parameters["rate2"]

        data_frame = fat_cross_df(
            arr_list=[
                MMOO(mu=mu1, lamb=lamb1, burst=burst1),
                MMOO(mu=mu2, lamb=lamb2, burst=burst2)
            ],
            ser_list=[ConstantRate(rate=rate1),
                      ConstantRate(rate=rate2)],
            opt_method=OptMethod.GRID_SEARCH,
            perform_param_list=perform_param_list)
    else:
        raise NameError("This arrival process is not implemented")

    for key, value in parameters.items():
        filename += "_" + key + "_" + str(value)

    data_frame.to_csv(
        filename + '.csv', index=True, quoting=csv.QUOTE_NONNUMERIC)

    return data_frame


if __name__ == '__main__':
    DELAY_PROB_LIST = PerformParamList(
        perform_metric=PerformMetric.DELAY_PROB, values_list=range(4, 11))
    EXP_ARRIVAL1 = ExponentialArrival()
    MMOO1 = MMOO()

    # print(
    #     csv_fat_cross_perform(
    #         arrival=EXP_ARRIVAL1,
    #         parameters={"lamb1": 0.2,
    #                     "lamb2": 10,
    #                     "rate1": 14,
    #                     "rate2": 0.2},
    #         perform_param_list=DELAY_PROB_LIST
    #         opt_method=OptMethod.GRID_SEARCH))
    #
    # print(
    #     csv_fat_cross_perform(
    #         arrival=EXP_ARRIVAL1,
    #         parameters={
    #             "lamb1": 0.6,
    #             "lamb2": 4.5,
    #             "rate1": 3.2,
    #             "rate2": 0.3
    #         },
    #         perform_param_list=DELAY_PROB_LIST,
    #         opt_method=OptMethod.GRID_SEARCH))

    # print(
    #     csv_fat_cross_perform(
    #         arrival=EXP_ARRIVAL1,
    #         parameters={
    #             "lamb1": 0.4,
    #             "lamb2": 3.5,
    #             "rate1": 4.5,
    #             "rate2": 0.4
    #         },
    #         perform_param_list=DELAY_PROB_LIST
    #         opt_method=OptMethod.GRID_SEARCH))

    print(
        csv_fat_cross_perform(
            arrival=MMOO1,
            parameters={
                "mu1": 1.0,
                "mu2": 3.6,
                "lamb1": 2.2,
                "lamb2": 1.6,
                "burst1": 3.4,
                "burst2": 0.4,
                "rate1": 2.0,
                "rate2": 0.3
            },
            perform_param_list=DELAY_PROB_LIST,
            opt_method=OptMethod.GRID_SEARCH))

    # print(
    #     csv_fat_cross_perform(
    #         arrival=MMOO1,
    #         parameters={
    #             "mu1": 1.1,
    #             "mu2": 8.0,
    #             "lamb1": 2.0,
    #             "lamb2": 1.0,
    #             "burst1": 8.0,
    #             "burst2": 1.0,
    #             "rate1": 5,
    #             "rate2": 0.9
    #         },
    #         perform_param_list=DELAY_PROB_LIST,
    #         opt_method=OptMethod.GRID_SEARCH))
