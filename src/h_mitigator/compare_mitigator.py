"""Compare standard standard_bound with h-mitigator."""

from math import nan
from timeit import default_timer as timer
from typing import Tuple

from h_mitigator.optimize_mitigator import OptimizeMitigator
from h_mitigator.setting_mitigator import SettingMitigator
from nc_operations.perform_enum import PerformEnum
from optimization.initial_simplex import InitialSimplex
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize


def compare_mitigator(setting: SettingMitigator,
                      opt_method: OptMethod,
                      number_l=1,
                      print_x=False) -> Tuple[float, float]:
    """Compare standard_bound with the new Lyapunov standard_bound."""

    if opt_method == OptMethod.GRID_SEARCH:
        theta_bounds = [(0.1, 4.0)]

        standard_bound = Optimize(setting=setting,
                                  number_param=1,
                                  print_x=print_x).grid_search(
                                      bound_list=theta_bounds, delta=0.1)

        bound_array = theta_bounds[:]
        for _i in range(1, number_l + 1):
            bound_array.append((0.9, 4.0))

        h_mit_bound = OptimizeMitigator(setting_h_mit=setting,
                                        number_param=number_l + 1,
                                        print_x=print_x).grid_search(
                                            bound_list=bound_array, delta=0.1)

    elif opt_method == OptMethod.PATTERN_SEARCH:
        theta_start = 0.5

        start_list = [theta_start]

        standard_bound = Optimize(setting=setting,
                                  number_param=1,
                                  print_x=print_x).pattern_search(
                                      start_list=start_list,
                                      delta=3.0,
                                      delta_min=0.01)

        start_list_new = [theta_start] + [1.0] * number_l

        h_mit_bound = OptimizeMitigator(setting_h_mit=setting,
                                        number_param=number_l + 1,
                                        print_x=print_x).pattern_search(
                                            start_list=start_list_new,
                                            delta=3.0,
                                            delta_min=0.01)

        # This part is there to overcome opt_method issues
        if h_mit_bound > standard_bound:
            h_mit_bound = standard_bound

    elif opt_method == OptMethod.NELDER_MEAD:
        theta_start = 0.5

        start_list = [theta_start]
        start_simplex = InitialSimplex(parameters_to_optimize=1).gao_han(
            start_list=start_list)

        standard_bound = Optimize(setting=setting,
                                  number_param=1,
                                  print_x=print_x).nelder_mead(
                                      simplex=start_simplex, sd_min=10**(-2))

        start_list_new = [theta_start] + [1.0] * number_l
        start_simplex_new = InitialSimplex(parameters_to_optimize=number_l +
                                           1).gao_han(
                                               start_list=start_list_new)

        h_mit_bound = OptimizeMitigator(setting_h_mit=setting,
                                        number_param=number_l + 1,
                                        print_x=print_x).nelder_mead(
                                            simplex=start_simplex_new,
                                            sd_min=10**(-2))

        # This part is there to overcome opt_method issues
        if h_mit_bound > standard_bound:
            h_mit_bound = standard_bound

    elif opt_method == OptMethod.BASIN_HOPPING:
        theta_start = 0.5

        start_list = [theta_start]

        standard_bound = Optimize(
            setting=setting, number_param=1,
            print_x=print_x).basin_hopping(start_list=start_list)

        start_list_new = [theta_start] + [1.0] * number_l

        h_mit_bound = OptimizeMitigator(
            setting_h_mit=setting, number_param=number_l + 1,
            print_x=print_x).basin_hopping(start_list=start_list_new)

        # This part is there to overcome opt_method issues
        if h_mit_bound > standard_bound:
            h_mit_bound = standard_bound

    elif opt_method == OptMethod.DUAL_ANNEALING:
        theta_bounds = [(0.1, 4.0)]

        standard_bound = Optimize(
            setting=setting, number_param=1,
            print_x=print_x).dual_annealing(bound_list=theta_bounds)

        bound_array = theta_bounds[:]
        for _i in range(1, number_l + 1):
            bound_array.append((0.9, 4.0))

        h_mit_bound = OptimizeMitigator(
            setting_h_mit=setting, number_param=number_l + 1,
            print_x=print_x).dual_annealing(bound_list=bound_array)

        # This part is there to overcome opt_method issues
        if h_mit_bound > standard_bound:
            h_mit_bound = standard_bound

    elif opt_method == OptMethod.DIFFERENTIAL_EVOLUTION:
        theta_bounds = [(0.1, 8.0)]

        standard_bound = Optimize(
            setting=setting, number_param=1,
            print_x=print_x).diff_evolution(bound_list=theta_bounds)

        bound_array = theta_bounds[:]
        for _i in range(1, number_l + 1):
            bound_array.append((0.9, 8.0))

        h_mit_bound = OptimizeMitigator(
            setting_h_mit=setting, number_param=number_l + 1,
            print_x=print_x).diff_evolution(bound_list=bound_array)

    else:
        raise NameError(
            f"Optimization parameter {opt_method.name} is infeasible")

    # This part is there to overcome opt_method issues
    if h_mit_bound > standard_bound:
        h_mit_bound = standard_bound

    if standard_bound == 0 or h_mit_bound == 0:
        standard_bound = nan
        h_mit_bound = nan

    return standard_bound, h_mit_bound


def compare_time(setting: SettingMitigator,
                 opt_method: OptMethod,
                 number_l=1) -> tuple:
    """Compare computation times."""

    if opt_method == OptMethod.GRID_SEARCH:
        bound_array = [(0.1, 4.0)]

        start = timer()
        Optimize(setting=setting,
                 number_param=1).grid_search(bound_list=bound_array, delta=0.1)
        stop = timer()
        time_standard = stop - start

        for _ in range(1, number_l + 1):
            bound_array.append((0.9, 4.0))

        start = timer()
        OptimizeMitigator(setting_h_mit=setting, number_param=number_l +
                          1).grid_search(bound_list=bound_array, delta=0.1)
        stop = timer()
        time_lyapunov = stop - start

    elif opt_method == OptMethod.PATTERN_SEARCH:
        start_list = [0.5]

        start = timer()
        Optimize(setting=setting,
                 number_param=1).pattern_search(start_list=start_list,
                                                delta=3.0,
                                                delta_min=0.01)
        stop = timer()
        time_standard = stop - start

        start_list = [0.5] + [1.0] * number_l

        start = timer()
        OptimizeMitigator(setting_h_mit=setting, number_param=number_l +
                          1).pattern_search(start_list=start_list,
                                            delta=3.0,
                                            delta_min=0.01)
        stop = timer()
        time_lyapunov = stop - start

    elif opt_method == OptMethod.NELDER_MEAD:
        start_simplex = InitialSimplex(parameters_to_optimize=1).uniform_dist(
            max_theta=1.0)

        start = timer()
        Optimize(setting=setting,
                 number_param=1).nelder_mead(simplex=start_simplex,
                                             sd_min=10**(-2))
        stop = timer()
        time_standard = stop - start

        start_simplex_new = InitialSimplex(parameters_to_optimize=number_l +
                                           1).uniform_dist(max_theta=1.0,
                                                           max_l=2.0)

        start = timer()
        OptimizeMitigator(setting_h_mit=setting, number_param=number_l +
                          1).nelder_mead(simplex=start_simplex_new,
                                         sd_min=10**(-2))
        stop = timer()
        time_lyapunov = stop - start

    elif opt_method == OptMethod.DUAL_ANNEALING:
        bound_array = [(0.1, 4.0)]

        start = timer()
        Optimize(setting=setting,
                 number_param=1).dual_annealing(bound_list=bound_array)
        stop = timer()
        time_standard = stop - start

        for _ in range(1, number_l + 1):
            bound_array.append((0.9, 4.0))

        start = timer()
        OptimizeMitigator(setting_h_mit=setting, number_param=number_l +
                          1).dual_annealing(bound_list=bound_array)
        stop = timer()
        time_lyapunov = stop - start

    else:
        raise NameError(
            f"Optimization parameter {opt_method.name} is infeasible")

    return time_standard, time_lyapunov


if __name__ == '__main__':
    from h_mitigator.fat_cross_perform import FatCrossPerform
    from h_mitigator.single_server_mit_perform import SingleServerMitPerform
    from utils.perform_parameter import PerformParameter
    from nc_server.constant_rate_server import ConstantRateServer
    from nc_arrivals.qt import DM1

    OUTPUT_TIME = PerformParameter(perform_metric=PerformEnum.OUTPUT, value=4)

    SETTING1 = SingleServerMitPerform(arr_list=[DM1(lamb=4.4)],
                                      server=ConstantRateServer(rate=0.24),
                                      perform_param=OUTPUT_TIME)

    # print(
    #     compare_mitigator(
    #         setting=SETTING1, opt_method=OptMethod.GRID_SEARCH,
    #         print_x=True))

    DELAY_PROB = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                  value=4)

    ARR_LIST = [DM1(lamb=11.0), DM1(lamb=9.0)]
    SER_LIST = [ConstantRateServer(rate=5.0), ConstantRateServer(rate=4.0)]

    SETTING2 = FatCrossPerform(arr_list=ARR_LIST,
                               ser_list=SER_LIST,
                               perform_param=DELAY_PROB)

    print("compare bounds\n")

    print(
        compare_mitigator(setting=SETTING2,
                          opt_method=OptMethod.GRID_SEARCH,
                          print_x=True))

    print(
        compare_mitigator(setting=SETTING2,
                          opt_method=OptMethod.PATTERN_SEARCH,
                          print_x=True))

    print(
        compare_mitigator(setting=SETTING2,
                          opt_method=OptMethod.DUAL_ANNEALING,
                          print_x=True))

    print("\ncompare runtimes\n")

    print(
        compare_time(setting=SETTING2,
                     opt_method=OptMethod.GRID_SEARCH,
                     number_l=1))

    print(
        compare_time(setting=SETTING2,
                     opt_method=OptMethod.DUAL_ANNEALING,
                     number_l=1))
