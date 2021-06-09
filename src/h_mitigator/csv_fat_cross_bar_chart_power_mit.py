"""Creates data for barplot to observe scaling for more servers."""

import csv
from typing import List
from warnings import warn

import numpy as np
import pandas as pd

from bound_evaluation.change_enum import ChangeEnum
from h_mitigator.compare_mitigator import compare_mitigator
from h_mitigator.fat_cross_perform import FatCrossPerform
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.iid import DM1
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.opt_method import OptMethod
from utils.perform_parameter import PerformParameter


def csv_bar_chart(ar_list: List[ArrivalDistribution],
                  ser_list: List[ConstantRateServer],
                  perform_param: PerformParameter,
                  opt_method: OptMethod,
                  change_metric=ChangeEnum.RATIO_REF_NEW) -> pd.DataFrame:
    """Write the data for a special case in a csv."""
    size = len(ar_list)

    bar_matrix = np.zeros(shape=[size, 4])
    ar_list_copy = ar_list[:]
    ser_list_copy = ser_list[:]

    for i in range(size - 1, 0, -1):
        # i = row list index = number_servers - 1. That's why i goes to 0,
        # not 1
        large_setting = FatCrossPerform(arr_list=ar_list_copy,
                                        ser_list=ser_list_copy,
                                        perform_param=perform_param)

        utilization = format(large_setting.approximate_utilization(), '.3f')

        standard_bound, h_mit_bound = compare_mitigator(setting=large_setting,
                                                        opt_method=opt_method,
                                                        number_l=i)

        standard_bound = float(format(standard_bound, '.4f'))
        h_mit_bound = float(format(h_mit_bound, '.4f'))

        if h_mit_bound >= 1:
            warn(f"h-mitigator bound = {h_mit_bound} is >= 1")

        if change_metric == ChangeEnum.RATIO_REF_NEW:
            # improvement factor
            opt_diff = float(format(standard_bound / h_mit_bound, '.4f'))

        elif change_metric == ChangeEnum.DIFF_REF_NEW:
            # absolute difference
            opt_diff = float(format(standard_bound - h_mit_bound, '.4f'))

        else:
            raise NameError(
                f"metric parameter = {change_metric} is infeasible")

        bar_matrix[i, ] = [standard_bound, h_mit_bound, opt_diff, utilization]

        del ar_list_copy[-1]
        del ser_list_copy[-1]
        # delete one flow and its server to analyze a smaller network

    delay_bounds_df = pd.DataFrame({
        "number_servers": range(1, size + 1),
        "standard_bound": bar_matrix[:, 0],
        "h_mit_bound": bar_matrix[:, 1],
        "improvement": bar_matrix[:, 2],
        "utilization": bar_matrix[:, 3],
    })

    delay_bounds_df = delay_bounds_df[[
        "number_servers", "standard_bound", "h_mit_bound", "improvement",
        "utilization"
    ]]

    delay_bounds_df.to_csv(
        f"bar_chart_{str(size)}_{opt_method.name}_improvement_factor.csv",
        index=True,
        quoting=csv.QUOTE_NONNUMERIC)

    return delay_bounds_df


if __name__ == '__main__':
    DELAY_PROB4 = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                   value=8)
    print(
        csv_bar_chart(ar_list=[
            DM1(lamb=0.5),
            DM1(lamb=8.0),
            DM1(lamb=8.0),
            DM1(lamb=8.0),
            DM1(lamb=8.0),
            DM1(lamb=8.0),
            DM1(lamb=8.0),
            DM1(lamb=8.0)
        ],
                      ser_list=[
                          ConstantRateServer(rate=4.0),
                          ConstantRateServer(rate=2.0),
                          ConstantRateServer(rate=2.0),
                          ConstantRateServer(rate=2.0),
                          ConstantRateServer(rate=2.0),
                          ConstantRateServer(rate=2.0),
                          ConstantRateServer(rate=2.0),
                          ConstantRateServer(rate=2.0)
                      ],
                      perform_param=DELAY_PROB4,
                      opt_method=OptMethod.PATTERN_SEARCH,
                      change_metric=ChangeEnum.RATIO_REF_NEW))
