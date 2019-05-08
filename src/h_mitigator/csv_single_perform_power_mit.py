"""Compute output bound and write into csv file"""

import csv

import pandas as pd

from h_mitigator.optimize_mitigator import OptimizeMitigator
from h_mitigator.single_server_mit_perform import SingleServerMitPerform
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.markov_modulated import MMOOFluid
from nc_arrivals.qt import DM1, MD1
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize
from utils.perform_param_list import PerformParamList

# import sys
# import os
# Necessary to make it executable in terminal
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
#                              os.pardir))


def single_server_df(arr1: ArrivalDistribution, ser1: ConstantRateServer,
                     opt_method: OptMethod,
                     perform_param_list: PerformParamList) -> pd.DataFrame:
    """
    Compute output bound for T in T_list and write into dataframe
    Args:
        arr1: Arrival object
        ser1: Service object
        opt_method: method name as string, GS or PS
        perform_param_list: list of performance parameter values

    Returns:
        dataframe
    """

    bound = [0.0] * len(perform_param_list)
    new_bound = [0.0] * len(perform_param_list)

    for _i in range(len(perform_param_list)):
        setting = SingleServerMitPerform(
            arr=arr1,
            const_rate=ser1,
            perform_param=perform_param_list.get_parameter_at_i(_i))

        if opt_method == OptMethod.GRID_SEARCH:
            bound[_i] = Optimize(setting=setting).grid_search(bound_list=[
                (0.1, 4.0)
            ],
                                                              delta=0.1)
            new_bound[_i] = OptimizeMitigator(
                setting_h_mit=setting).grid_search(bound_list=[(0.1, 4.0),
                                                               (0.9, 8.0)],
                                                   delta=0.05)

        elif opt_method == OptMethod.PATTERN_SEARCH:
            bound[_i] = Optimize(setting=setting).pattern_search(
                start_list=[0.5], delta=3.0, delta_min=0.01)
            new_bound[_i] = OptimizeMitigator(
                setting_h_mit=setting).pattern_search(start_list=[0.5, 2.0],
                                                      delta=3.0,
                                                      delta_min=0.01)
        else:
            raise NameError(
                f"Optimization parameter {opt_method} is infeasible")

    delay_bounds_df = pd.DataFrame({
        "bound": bound,
        "h_mit_bound": new_bound
    },
                                   index=perform_param_list.values_list)
    delay_bounds_df = delay_bounds_df[["bound", "h_mit_bound"]]

    return delay_bounds_df


def csv_single_perform(arrival: ArrivalDistribution,
                       service: ConstantRateServer,
                       perform_param_list: PerformParamList,
                       opt_method: OptMethod) -> pd.DataFrame:
    """
    Writes dataframe results into a csv file

    Args:
        arrival: flow of interest's arrival distribution
        service: service of the server at the foi
        perform_param_list: list of performance parameter values
        opt_method: optimization method

    Returns:
        csv file
    """

    filename = f"single_{perform_param_list.to_name()}"

    data_frame = single_server_df(arr1=arrival,
                                  ser1=service,
                                  opt_method=opt_method,
                                  perform_param_list=perform_param_list)

    filename += "_" + arrival.to_name() + "_" + arrival.to_value(
        number=1, show_n=False) + "_" + service.to_value(number=1)

    data_frame.to_csv(filename + '.csv',
                      index=True,
                      quoting=csv.QUOTE_NONNUMERIC)

    return data_frame


if __name__ == '__main__':
    OUTPUT_LIST = PerformParamList(perform_metric=PerformEnum.OUTPUT,
                                   values_list=range(4, 15))

    print(
        csv_single_perform(arrival=DM1(lamb=3.8, n=1),
                           service=ConstantRateServer(rate=3.0),
                           perform_param_list=OUTPUT_LIST,
                           opt_method=OptMethod.GRID_SEARCH))

    print(
        csv_single_perform(arrival=MMOOFluid(mu=8.0, lamb=12.0, burst=3.0,
                                             n=1),
                           service=ConstantRateServer(rate=1.5),
                           perform_param_list=OUTPUT_LIST,
                           opt_method=OptMethod.GRID_SEARCH))

    RATE_1 = ConstantRateServer(rate=1.0)

    print(
        csv_single_perform(arrival=MD1(lamb=0.5, mu=1.0),
                           service=RATE_1,
                           perform_param_list=OUTPUT_LIST,
                           opt_method=OptMethod.GRID_SEARCH))
