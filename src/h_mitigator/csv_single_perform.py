"""Compute output bound and write into csv file"""

import csv

import pandas as pd

from h_mitigator.single_server_mit_perform import SingleServerMitPerform
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.qt import MD1, MM1
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize
from utils.perform_param_list import PerformParamList


def single_server_df(arr1: ArrivalDistribution, ser1: ConstantRateServer,
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

    bound = [0.0] * len(perform_param_list)

    for i in range(len(perform_param_list)):
        setting = SingleServerMitPerform(
            arr=arr1,
            const_rate=ser1,
            perform_param=perform_param_list.get_parameter_at_i(i))

        if opt_method == OptMethod.GRID_SEARCH:
            bound[i] = Optimize(setting=setting).grid_search(bound_list=[
                (0.1, 4.0)
            ],
                                                             delta=0.1)

        elif opt_method == OptMethod.PATTERN_SEARCH:
            bound[i] = Optimize(setting=setting).pattern_search(
                start_list=[0.5], delta=3.0, delta_min=0.01)
        else:
            raise NameError(
                "Optimization parameter {0} is infeasible".format(opt_method))

    delay_bounds_df = pd.DataFrame({"bound": bound},
                                   index=perform_param_list.values_list)
    delay_bounds_df = delay_bounds_df[["bound"]]

    return delay_bounds_df


def csv_single_perform(arrival: ArrivalDistribution,
                       service: ConstantRateServer,
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
    DELAY_LIST = PerformParamList(perform_metric=PerformEnum.DELAY_PROB,
                                  values_list=range(15, 41))
    CONST1 = ConstantRateServer(rate=1.0)

    print(
        csv_single_perform(arrival=MD1(lamb=0.8, mu=1.0),
                           service=CONST1,
                           perform_param_list=DELAY_LIST,
                           opt_method=OptMethod.GRID_SEARCH))

    print(
        csv_single_perform(arrival=MM1(lamb=0.8, mu=1.0),
                           service=CONST1,
                           perform_param_list=DELAY_LIST,
                           opt_method=OptMethod.GRID_SEARCH))
