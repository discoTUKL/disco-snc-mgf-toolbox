"""Compute output bound for various delta times and write into csv file"""

import csv
from typing import List

import pandas as pd

from library.perform_param_list import PerformParamList
from nc_operations.perform_metric import PerformMetric
from nc_processes.arrival_distribution import (MMOO, ArrivalDistribution,
                                               ExponentialArrival)
from nc_processes.distrib_param import DistribParam
from nc_processes.service import ConstantRate, Service
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize
from optimization.optimize_new import OptimizeNew
from single_server.single_server_perform import SingleServerPerform

# import sys
# import os
# Necessary to make it executable in terminal
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
#                              os.pardir))


def single_server_df(arr1: ArrivalDistribution, ser1: Service,
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
            ser=ser1,
            perform_param=perform_param_list.get_parameter_at_i(_i))

        if opt_method == OptMethod.GRID_SEARCH:
            bound[_i] = Optimize(setting=setting).grid_search_old(
                bound_list=[(0.1, 4.0)], delta=0.1)
            new_bound[_i] = OptimizeNew(
                setting_new=setting, new=True).grid_search_old(
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


def csv_single_perform(arrival: ArrivalDistribution,
                       list_of_parameters: List[DistribParam],
                       perform_param_list: PerformParamList,
                       opt_method: OptMethod) -> pd.DataFrame:
    """Writes dataframe results into a csv file

    Args:
        arrival: String that represents the arrival process
        list_of_parameters: dictionaries of actual values
        perform_param_list: list of performance parameter values
        opt_method: optimization method

    Returns:
        csv file
    """

    filename = "single_{0}".format(perform_param_list.perform_metric.name)

    if isinstance(arrival, ExponentialArrival):
        arr1 = ExponentialArrival(list_of_parameters[0].lamb)
        filename += list_of_parameters[0].get_exp_string(1)
    elif isinstance(arrival, MMOO):
        arr1 = MMOO(list_of_parameters[0].mu, list_of_parameters[0].lamb,
                    list_of_parameters[0].burst)
        filename += list_of_parameters[0].get_mmoo_string(1)
    else:
        raise NameError("This arrival process is not implemented")

    rate1 = list_of_parameters[1].rate
    filename += list_of_parameters[1].get_constant_string(1)

    data_frame = single_server_df(
        arr1=arr1,
        ser1=ConstantRate(rate1),
        opt_method=opt_method,
        perform_param_list=perform_param_list)

    data_frame.to_csv(
        filename + '.csv', index=True, quoting=csv.QUOTE_NONNUMERIC)

    return data_frame


if __name__ == '__main__':
    OUTPUT_LIST = PerformParamList(
        perform_metric=PerformMetric.OUTPUT, values_list=range(4, 15))
    EXP_ARRIVAL1 = ExponentialArrival()
    MMOO_ARRIVAL = MMOO()

    exp1 = DistribParam(lamb=1.0)
    mmoo1 = DistribParam(mu=8.0, lamb=12.0, burst=3.0)
    const_rate1 = DistribParam(rate=2.0)
    const_rate2 = DistribParam(rate=1.3)

    NEW_OUTPUT_LIST = PerformParamList(
        perform_metric=PerformMetric.OUTPUT, values_list=range(4, 15))

    parameters_exp = [exp1, const_rate1]
    parameters_mmoo = [mmoo1, const_rate2]

    print(
        csv_single_perform(
            arrival=EXP_ARRIVAL1,
            list_of_parameters=parameters_exp,
            perform_param_list=NEW_OUTPUT_LIST,
            opt_method=OptMethod.GRID_SEARCH))

    print(
        csv_single_perform(
            arrival=MMOO_ARRIVAL,
            list_of_parameters=parameters_mmoo,
            perform_param_list=NEW_OUTPUT_LIST,
            opt_method=OptMethod.GRID_SEARCH))
