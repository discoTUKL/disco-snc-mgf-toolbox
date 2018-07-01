"""Compute output bound and write into csv file"""

import csv
import pandas as pd

from library.perform_param_list import PerformParamList
from nc_operations.perform_enum import PerformEnum
from nc_processes.arrival_distribution import (MMOO, ArrivalDistribution,
                                               ExponentialArrival)
from nc_processes.constant_rate_server import ConstantRate
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize
from optimization.optimize_new import OptimizeNew
from single_server.single_server_perform import SingleServerPerform

# import sys
# import os
# Necessary to make it executable in terminal
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
#                              os.pardir))


def single_server_df(arr1: ArrivalDistribution, ser1: ConstantRate,
                     opt_method: OptMethod,
                     perform_param_list: PerformParamList) -> pd.DataFrame:
    """Compute output bound for T in T_list and write into dataframe
    Args:
        arr1: Arrival object
        ser1: Service object
        opt_method: method name as string, GS or PS
        perform_param_list: list of performance parameter values

    Returns:
        dataframe
    """

    bound = [0.0] * len(perform_param_list.values_list)
    new_bound = [0.0] * len(perform_param_list.values_list)

    for _i in range(len(perform_param_list.values_list)):
        setting = SingleServerPerform(
            arr=arr1,
            const_rate=ser1,
            perform_param=perform_param_list.get_parameter_at_i(_i))

        if opt_method == OptMethod.GRID_SEARCH:
            bound[_i] = Optimize(setting=setting).grid_search(
                bound_list=[(0.1, 4.0)], delta=0.1)
            new_bound[_i] = OptimizeNew(
                setting_new=setting, new=True).grid_search(
                    bound_list=[(0.1, 4.0), (0.9, 8.0)], delta=0.1)

        elif opt_method == OptMethod.PATTERN_SEARCH:
            bound[_i] = Optimize(setting=setting).pattern_search(
                start_list=[0.5], delta=3.0, delta_min=0.01)
            new_bound[_i] = OptimizeNew(
                setting_new=setting, new=True).pattern_search(
                    start_list=[0.5, 2.0], delta=3.0, delta_min=0.01)
        else:
            raise NameError(
                "Optimization parameter {0} is infeasible".format(opt_method))

    delay_bounds_df = pd.DataFrame(
        {
            "bound": bound,
            "new_bound": new_bound
        },
        index=perform_param_list.values_list)
    delay_bounds_df = delay_bounds_df[["bound", "new_bound"]]

    return delay_bounds_df


def csv_single_perform(arrival: ArrivalDistribution, service: ConstantRate,
                       perform_param_list: PerformParamList,
                       opt_method: OptMethod) -> pd.DataFrame:
    """Writes dataframe results into a csv file

    Args:
        arrival: flow of interest's arrival distribution
        service: service of the server at the foi
        perform_param_list: list of performance parameter values
        opt_method: optimization method

    Returns:
        csv file
    """

    filename = "single_{0}".format(perform_param_list.to_name())

    data_frame = single_server_df(
        arr1=arrival,
        ser1=service,
        opt_method=opt_method,
        perform_param_list=perform_param_list)

    filename += "_" + arrival.to_value() + "_" + service.to_value()

    data_frame.to_csv(
        filename + '.csv', index=True, quoting=csv.QUOTE_NONNUMERIC)

    return data_frame


if __name__ == '__main__':
    OUTPUT_LIST = PerformParamList(
        perform_metric=PerformEnum.OUTPUT, values_list=range(4, 15))

    exp1 = ExponentialArrival(lamb=1.0)
    mmoo1 = MMOO(mu=8.0, lamb=12.0, burst=3.0)
    const_rate1 = ConstantRate(rate=2.0)
    const_rate2 = ConstantRate(rate=1.3)

    NEW_OUTPUT_LIST = PerformParamList(
        perform_metric=PerformEnum.OUTPUT, values_list=range(4, 15))

    print(
        csv_single_perform(
            arrival=exp1,
            service=const_rate1,
            perform_param_list=NEW_OUTPUT_LIST,
            opt_method=OptMethod.GRID_SEARCH))

    print(
        csv_single_perform(
            arrival=mmoo1,
            service=const_rate1,
            perform_param_list=NEW_OUTPUT_LIST,
            opt_method=OptMethod.GRID_SEARCH))
