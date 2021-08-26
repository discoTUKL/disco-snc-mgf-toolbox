"""Compare standard standard_bound with negative dependence."""

from timeit import default_timer as timer
from typing import Tuple

from nc_operations.perform_enum import PerformEnum
from optimization.optimize import Optimize

from msob_and_fp.optimize_fp_bound import OptimizeFPBound
from msob_and_fp.optimize_server_bound import OptimizeServerBound
from msob_and_fp.setting_avoid_dep import SettingMSOBFP


def compare_avoid_dep_211(setting: SettingMSOBFP,
                          print_x=False) -> Tuple[float, float, float]:
    """Compare standard_bound with the new Lyapunov standard_bound."""

    delta_val = 0.05

    one_param_bounds = [(0.1, 10.0)]
    two_param_bounds = [(0.1, 10.0), (1.1, 10.0)]

    standard_bound = Optimize(setting=setting, number_param=2,
                              print_x=print_x).grid_search(
                                  grid_bounds=two_param_bounds, delta=delta_val)

    server_bound = OptimizeServerBound(setting_msob_fp=setting,
                                       number_param=1,
                                       print_x=print_x).grid_search(
                                           grid_bounds=one_param_bounds,
                                           delta=delta_val)

    fp_bound = OptimizeFPBound(setting_msob_fp=setting,
                               number_param=1,
                               print_x=print_x).grid_search(
                                   grid_bounds=one_param_bounds,
                                   delta=delta_val)

    return standard_bound, server_bound, fp_bound


def compare_avoid_dep_212(setting: SettingMSOBFP,
                          print_x=False) -> Tuple[float, float, float]:
    """Compare standard_bound with the new Lyapunov standard_bound."""

    delta_val = 0.05

    one_param_bounds = [(0.1, 10.0)]
    two_param_bounds = [(0.1, 10.0), (1.1, 10.0)]

    standard_bound = Optimize(setting=setting, number_param=2,
                              print_x=print_x).grid_search(
                                  grid_bounds=two_param_bounds, delta=delta_val)

    server_bound = OptimizeServerBound(setting_msob_fp=setting,
                                       number_param=1,
                                       print_x=print_x).grid_search(
                                           grid_bounds=one_param_bounds,
                                           delta=delta_val)

    fp_bound = OptimizeFPBound(setting_msob_fp=setting,
                               number_param=2,
                               print_x=print_x).grid_search(
                                   grid_bounds=two_param_bounds,
                                   delta=delta_val)

    return standard_bound, server_bound, fp_bound


def compare_time_211(setting: SettingMSOBFP) -> Tuple[float, float, float]:
    """Compare standard_bound with the new Lyapunov standard_bound."""

    delta_val = 0.05

    one_param_bounds = [(0.1, 10.0)]
    two_param_bounds = [(0.1, 10.0), (1.1, 10.0)]

    start = timer()
    Optimize(setting=setting,
             number_param=2).grid_search(grid_bounds=two_param_bounds,
                                         delta=delta_val)
    stop = timer()
    time_standard = stop - start

    start = timer()
    OptimizeServerBound(setting_msob_fp=setting, number_param=1).grid_search(
        grid_bounds=one_param_bounds, delta=delta_val)
    stop = timer()
    time_server_bound = stop - start

    start = timer()
    OptimizeFPBound(setting_msob_fp=setting,
                    number_param=1).grid_search(grid_bounds=one_param_bounds,
                                                delta=delta_val)
    stop = timer()
    time_fp_bound = stop - start

    return time_standard, time_server_bound, time_fp_bound


def compare_time_212(setting: SettingMSOBFP) -> Tuple[float, float, float]:
    """Compare standard_bound with the new Lyapunov standard_bound."""

    delta_val = 0.05

    one_param_bounds = [(0.1, 10.0)]
    two_param_bounds = [(0.1, 10.0), (1.1, 10.0)]

    start = timer()
    Optimize(setting=setting,
             number_param=2).grid_search(grid_bounds=two_param_bounds,
                                         delta=delta_val)
    stop = timer()
    time_standard = stop - start

    start = timer()
    OptimizeServerBound(setting_msob_fp=setting, number_param=1).grid_search(
        grid_bounds=one_param_bounds, delta=delta_val)
    stop = timer()
    time_server_bound = stop - start

    start = timer()
    OptimizeFPBound(setting_msob_fp=setting,
                    number_param=2).grid_search(grid_bounds=two_param_bounds,
                                                delta=delta_val)
    stop = timer()
    time_fp_bound = stop - start

    return time_standard, time_server_bound, time_fp_bound


if __name__ == '__main__':
    from nc_arrivals.iid import DM1
    from nc_server.constant_rate_server import ConstantRateServer
    from utils.perform_parameter import PerformParameter

    from msob_and_fp.overlapping_tandem_perform import OverlappingTandemPerform

    DELAY_PROB = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                  value=10)

    ARR_LIST = [DM1(lamb=7.0), DM1(lamb=7.0), DM1(lamb=6.0)]

    SER_LIST = [
        ConstantRateServer(rate=0.5),
        ConstantRateServer(rate=8.0),
        ConstantRateServer(rate=5.5)
    ]

    SETTING = OverlappingTandemPerform(arr_list=ARR_LIST,
                                       ser_list=SER_LIST,
                                       perform_param=DELAY_PROB)

    # print(compare_avoid_dep_211(setting=SETTING, print_x=False))
    print(compare_time_211(setting=SETTING))
