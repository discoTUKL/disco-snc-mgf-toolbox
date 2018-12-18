"""Computed some examples from the project"""

from fat_tree.fat_cross_perform import FatCrossPerform
from nc_arrivals.qt import DM1
from nc_operations.perform_enum import PerformEnum
from nc_service.constant_rate_server import ConstantRate
from optimization.optimize import Optimize
from utils.perform_parameter import PerformParameter

if __name__ == '__main__':
    PROB_VALUES = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]

    for p in PROB_VALUES:
        DELAY_TIME = PerformParameter(
            perform_metric=PerformEnum.DELAY, value=p)

        EXAMPLE = FatCrossPerform(
            arr_list=[DM1(lamb=1.0)],
            ser_list=[ConstantRate(rate=2.0)],
            perform_param=DELAY_TIME)

        print(
            Optimize(setting=EXAMPLE, print_x=False,
                     show_warn=True).grid_search(
                         bound_list=[(0.01, 1.1)], delta=0.01))
