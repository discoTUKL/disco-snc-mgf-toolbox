"""Creates data for barplot to observe scaling for more servers."""

import csv
from typing import List

import numpy as np
import pandas as pd

from fat_tree.fat_cross_perform import FatCrossPerform
from library.compare_old_new import compute_improvement
from library.perform_parameter import PerformParameter
from nc_operations.perform_enum import PerformEnum
from nc_processes.arrival_distribution import (ArrivalDistribution,
                                               ExponentialArrival)
from nc_processes.constant_rate_server import ConstantRate
from optimization.opt_method import OptMethod


def csv_bar_chart(ar_list: List[ArrivalDistribution],
                  ser_list: List[ConstantRate],
                  perform_param: PerformParameter,
                  opt_method: OptMethod,
                  metric="relative") -> pd.DataFrame:
    """Write the data for a special case in a csv."""
    size = len(ar_list)

    bar_matrix = np.zeros(shape=[size, 3])
    ar_list_copy = ar_list[:]
    ser_list_copy = ser_list[:]

    for i in range(size - 1, 0, -1):
        # i = row list index = number_servers - 1. That's why i goes to 0,
        # not 1
        large_setting = FatCrossPerform(
            arr_list=ar_list_copy,
            ser_list=ser_list_copy,
            perform_param=perform_param)

        standard_bound, new_bound = compute_improvement(
            setting=large_setting, opt_method=opt_method, number_l=i)

        if metric == "relative":
            # improvement factor
            opt_diff = standard_bound / new_bound

        elif metric == "absolute":
            # absolute difference
            opt_diff = standard_bound - new_bound

        else:
            raise NameError(
                "metric parameter {0} is infeasible".format(metric))

        bar_matrix[i, ] = [standard_bound, new_bound, opt_diff]

        del ar_list_copy[-1]
        del ser_list_copy[-1]
        # delete one flow and its server to analyze a smaller network

    delay_bounds_df = pd.DataFrame({
        "number_servers": range(1, size + 1),
        "standard_bound": bar_matrix[:, 0],
        "new_bound": bar_matrix[:, 1],
        "improvement": bar_matrix[:, 2]
    })

    delay_bounds_df = delay_bounds_df[[
        "number_servers", "standard_bound", "new_bound", "improvement"
    ]]

    delay_bounds_df.to_csv(
        'bar_chart_{0}_{1}_improvement_factor.csv'.format(
            str(size), opt_method.name),
        index=True,
        quoting=csv.QUOTE_NONNUMERIC)

    return delay_bounds_df


if __name__ == '__main__':
    DELAY_PROB4 = PerformParameter(
        perform_metric=PerformEnum.DELAY_PROB, value=4)
    print(
        csv_bar_chart(
            ar_list=[
                ExponentialArrival(lamb=0.5),
                ExponentialArrival(lamb=8.0),
                ExponentialArrival(lamb=8.0),
                ExponentialArrival(lamb=8.0),
                ExponentialArrival(lamb=8.0),
                ExponentialArrival(lamb=8.0),
                ExponentialArrival(lamb=8.0),
                ExponentialArrival(lamb=8.0),
                ExponentialArrival(lamb=8.0)
            ],
            ser_list=[
                ConstantRate(rate=4.5),
                ConstantRate(rate=2.0),
                ConstantRate(rate=2.0),
                ConstantRate(rate=2.0),
                ConstantRate(rate=2.0),
                ConstantRate(rate=2.0),
                ConstantRate(rate=2.0),
                ConstantRate(rate=2.0),
                ConstantRate(rate=2.0)
            ],
            perform_param=DELAY_PROB4,
            opt_method=OptMethod.PATTERN_SEARCH,
            metric="relative"))
