"""Writes a data frame of arrivals, service, and performance parameters
    into a csv"""

import csv
from typing import Callable, List

import pandas as pd

from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_server.constant_rate_server import ConstantRateServer
from optimization.opt_method import OptMethod
from utils.perform_param_list import PerformParamList


def data_frame_to_csv(prefix: str,
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
