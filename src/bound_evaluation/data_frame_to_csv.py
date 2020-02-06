"""Writes a data frame of arrivals, service, and performance parameters
    into a csv"""

import csv
from typing import Callable, List

import pandas as pd

from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_server.constant_rate_server import ConstantRateServer
from optimization.opt_method import OptMethod
from utils.perform_parameter import PerformParameter
from utils.perform_param_list import PerformParamList


def perform_param_list_to_csv(prefix: str,
                              data_frame_creator: Callable,
                              arr_list: List[ArrivalDistribution],
                              ser_list: List[ConstantRateServer],
                              perform_param_list: PerformParamList,
                              opt_method: OptMethod = None,
                              suffix="") -> pd.DataFrame:
    filename = prefix + perform_param_list.to_name()

    if opt_method is None:
        data_frame = data_frame_creator(arr_list=arr_list,
                                        ser_list=ser_list,
                                        perform_param_list=perform_param_list)
    else:
        data_frame = data_frame_creator(arr_list=arr_list,
                                        ser_list=ser_list,
                                        opt_method=opt_method,
                                        perform_param_list=perform_param_list)

    filename += "_" + arr_list[0].to_name()
    for i in range(len(arr_list)):
        filename += "_"
        filename += arr_list[i].to_value(number=i + 1)
    for j in range(len(ser_list)):
        filename += "_"
        filename += ser_list[j].to_value(number=j + 1)

    data_frame.to_csv(filename + suffix + '.csv',
                      index=True,
                      quoting=csv.QUOTE_NONNUMERIC)

    return data_frame


def arrival_list_to_csv(prefix: str,
                        data_frame_creator: Callable,
                        list_arr_list: List[List[ArrivalDistribution]],
                        ser_list: List[ConstantRateServer],
                        server_index: int,
                        perform_param: PerformParameter,
                        opt_method: OptMethod = None,
                        suffix="_adjusting_arrivals") -> pd.DataFrame:
    filename = prefix + perform_param.to_name()

    if opt_method is None:
        data_frame: pd.DataFrame = data_frame_creator(
            list_arr_list=list_arr_list,
            ser_list=ser_list,
            server_index=server_index,
            perform_param=perform_param)
    else:
        data_frame: pd.DataFrame = data_frame_creator(
            list_arr_list=list_arr_list,
            ser_list=ser_list,
            server_index=server_index,
            opt_method=opt_method,
            perform_param=perform_param)

    filename += "_" + list_arr_list[0][0].to_name()

    data_frame.to_csv(filename + suffix + '.csv',
                      index=True,
                      quoting=csv.QUOTE_NONNUMERIC)

    return data_frame
