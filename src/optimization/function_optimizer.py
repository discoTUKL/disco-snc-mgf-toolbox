"""Simple Optimizer"""

from math import inf
from typing import List

import numpy as np
import scipy.optimize

from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_server.constant_rate_server import ConstantRateServer
from utils.exceptions import ParameterOutOfBounds
from utils.perform_parameter import PerformParameter


def optimizer_perform(fun: callable, arr_list: List[ArrivalDistribution],
                      ser_list: List[ConstantRateServer],
                      perform_param: PerformParameter, ranges: list,
                      print_x=False) -> float:
    def helper_fun(param_list: [float, float]) -> float:
        try:
            return fun(param_list=param_list,
                       perform_param=perform_param,
                       arr_list=arr_list,
                       ser_list=ser_list)
        except (FloatingPointError, OverflowError, ParameterOutOfBounds,
                ValueError):
            return inf

    # np.seterr("raise")
    np.seterr("warn")

    grid_res = scipy.optimize.brute(func=helper_fun,
                                    ranges=ranges,
                                    full_output=True)

    if print_x:
        print("grid search optimal x: ", grid_res[0].tolist())

    return grid_res[1]
