"""Compute performance standard_bound and write into csv file"""

from typing import List

import pandas as pd

from bound_evaluation.data_frame_to_csv import perform_param_list_to_csv
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.iid import MD1, MM1
from nc_operations.perform_enum import PerformEnum
from nc_operations.single_server_perform import SingleServerPerform
from nc_server.constant_rate_server import ConstantRateServer
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize
from utils.perform_param_list import PerformParamList


def single_server_df(arr_list: List[ArrivalDistribution],
                     ser_list: List[ConstantRateServer], opt_method: OptMethod,
                     perform_param_list: PerformParamList) -> pd.DataFrame:
    """Compute output standard_bound for T in T_list and write into dataframe
    Args:
        arr_list: Arrival object list
        ser_list: Service object list
        opt_method: method name as string, GS or PS
        perform_param_list: list of performance parameter values

    Returns:
        dataframe
    """

    bound = [0.0] * len(perform_param_list)

    for i in range(len(perform_param_list)):
        setting = SingleServerPerform(
            foi=arr_list[0],
            server=ser_list[0],
            perform_param=perform_param_list.get_parameter_at_i(i))

        if opt_method == OptMethod.GRID_SEARCH:
            bound[i] = Optimize(setting=setting, number_param=1).grid_search(
                grid_bounds=[(0.1, 4.0)], delta=0.1).obj_value

        elif opt_method == OptMethod.PATTERN_SEARCH:
            bound[i] = Optimize(setting=setting,
                                number_param=1).pattern_search(
                                    start_list=[0.5],
                                    delta=3.0,
                                    delta_min=0.01).obj_value
        else:
            raise NameError(
                f"Optimization parameter {opt_method} is infeasible")

    delay_bounds_df = pd.DataFrame({"standard_bound": bound},
                                   index=perform_param_list.values_list)
    delay_bounds_df = delay_bounds_df[["standard_bound"]]

    return delay_bounds_df


if __name__ == '__main__':
    DELAY_LIST = PerformParamList(perform_metric=PerformEnum.DELAY_PROB,
                                  values_list=list(range(15, 41)))
    CONST1 = ConstantRateServer(rate=1.0)

    print(
        perform_param_list_to_csv(prefix="single_",
                                  data_frame_creator=single_server_df,
                                  arr_list=[MD1(lamb=0.8, mu=1.0)],
                                  ser_list=[CONST1],
                                  perform_param_list=DELAY_LIST,
                                  opt_method=OptMethod.GRID_SEARCH))

    print(
        perform_param_list_to_csv(prefix="single_",
                                  data_frame_creator=single_server_df,
                                  arr_list=[MM1(lamb=0.8, mu=1.0)],
                                  ser_list=[CONST1],
                                  perform_param_list=DELAY_LIST,
                                  opt_method=OptMethod.GRID_SEARCH))
