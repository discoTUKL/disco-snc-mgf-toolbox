"""Small examples to play with."""

from typing import List

import numpy as np
import scipy.optimize

from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.markov_modulated import MMOOFluid
from nc_operations.perform_enum import PerformEnum
from nc_operations.single_server_bandwidth import SingleServerBandwidth
from nc_server.constant_rate_server import ConstantRateServer
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize
from utils.perform_parameter import PerformParameter


def get_bandwidth_from_delay(arr_list: List[ArrivalDistribution],
                             target_delay: int,
                             target_delay_prob: float,
                             lower_interval: float,
                             upper_interval: float,
                             indep=True,
                             opt_method=OptMethod.GRID_SEARCH,
                             geom_series=True) -> float:
    def helper_function(rate: float):
        if opt_method == OptMethod.GRID_SEARCH:
            if indep:
                single_server = SingleServerBandwidth(
                    arr_list=arr_list,
                    s_e2e=ConstantRateServer(rate=rate),
                    perform_param=PerformParameter(
                        perform_metric=PerformEnum.DELAY_PROB,
                        value=target_delay),
                    indep=True,
                    geom_series=geom_series)

                current_delay_prob = Optimize(setting=single_server,
                                              number_param=1).grid_search(
                                                  grid_bounds=[(0.1, 5.0)],
                                                  delta=0.1)
            else:
                single_server = SingleServerBandwidth(
                    arr_list=arr_list,
                    s_e2e=ConstantRateServer(rate=rate),
                    perform_param=PerformParameter(
                        perform_metric=PerformEnum.DELAY_PROB,
                        value=target_delay),
                    indep=False,
                    geom_series=geom_series)

                current_delay_prob = Optimize(setting=single_server,
                                              number_param=2).grid_search(
                                                  grid_bounds=[(0.1, 5.0),
                                                               (1.1, 5.0)],
                                                  delta=0.1)

        else:
            raise NotImplementedError("This optimization method is not "
                                      "implemented")

        return current_delay_prob - target_delay_prob

    # np.seterr("raise")
    np.seterr("warn")

    res = scipy.optimize.bisect(helper_function,
                                a=lower_interval,
                                b=upper_interval,
                                full_output=True)
    return res[0]


if __name__ == '__main__':
    print("Single Server Performance Bounds:\n")

    DELAY6 = PerformParameter(perform_metric=PerformEnum.DELAY_PROB, value=6)

    ARR_LIST = [MMOOFluid(mu=0.2, lamb=0.5, peak_rate=2.6)]

    SINGLE_SERVER = SingleServerBandwidth(arr_list=ARR_LIST,
                                          s_e2e=ConstantRateServer(rate=2.0),
                                          perform_param=DELAY6)
    RESULTING_DELAY_PROB = Optimize(SINGLE_SERVER,
                                    number_param=1,
                                    print_x=True).grid_search(grid_bounds=[
                                        (0.1, 5.0)
                                    ],
                                                              delta=0.1)
    print(f"delay probability = {RESULTING_DELAY_PROB}")

    REQUIRED_BANDWIDTH = get_bandwidth_from_delay(arr_list=ARR_LIST,
                                                  target_delay=6,
                                                  target_delay_prob=0.034,
                                                  lower_interval=0.0,
                                                  upper_interval=200.0)
    print(f"required bandwidth = {REQUIRED_BANDWIDTH}")
