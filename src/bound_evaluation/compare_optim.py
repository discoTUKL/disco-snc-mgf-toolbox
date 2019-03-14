"""We compare the different optimizations"""

from timeit import default_timer as timer
from typing import List

from optimization.initial_simplex import InitialSimplex
from optimization.nelder_mead_parameters import NelderMeadParameters
from optimization.opt_method import OptMethod
from h_mitigator.optimize_mitigator import OptimizeMitigator
from optimization.sim_anneal_param import SimAnnealParams
from h_mitigator.setting_mitigator import SettingMitigator


def compare_optimization(setting: SettingMitigator,
                         opt_methods: List[OptMethod],
                         number_l=1) -> List[float]:
    """Measures time for different optimizations"""
    print_x = False

    list_of_bounds: List[float] = []
    list_of_times: List[float] = []
    list_of_approaches: List[str] = []

    for opt in opt_methods:
        start = timer()
        if opt == OptMethod.GRID_SEARCH:
            theta_bounds = [(0.1, 4.0)]

            bound_list = theta_bounds[:]
            for _i in range(1, number_l + 1):
                bound_list.append((0.9, 4.0))

            bound = OptimizeMitigator(
                setting_h_mit=setting, print_x=print_x).grid_search(
                    bound_list=bound_list, delta=0.1)

        elif opt == OptMethod.PATTERN_SEARCH:
            theta_start = 0.5

            start_list = [theta_start] + [1.0] * number_l

            bound = OptimizeMitigator(
                setting_h_mit=setting, print_x=print_x).pattern_search(
                    start_list=start_list, delta=3.0, delta_min=0.01)

        elif opt == OptMethod.NELDER_MEAD:
            theta_start = 0.5

            start_list = [theta_start] + [1.0] * number_l
            start_simplex = InitialSimplex(
                parameters_to_optimize=number_l + 1).gao_han(
                    start_list=start_list)

            bound = OptimizeMitigator(
                setting_h_mit=setting, print_x=print_x).nelder_mead(
                    simplex=start_simplex, sd_min=10**(-2))

        elif opt == OptMethod.BASIN_HOPPING:
            theta_start = 0.5

            start_list = [theta_start] + [1.0] * number_l

            bound = OptimizeMitigator(
                setting_h_mit=setting,
                print_x=print_x).basin_hopping(start_list=start_list)

        elif opt == OptMethod.SIMULATED_ANNEALING:
            simul_anneal_param = SimAnnealParams()
            theta_start = 0.5

            start_list = [theta_start] + [1.0] * number_l

            bound = OptimizeMitigator(
                setting_h_mit=setting, print_x=print_x).sim_annealing(
                    start_list=start_list,
                    sim_anneal_params=simul_anneal_param)

        elif opt == OptMethod.DIFFERENTIAL_EVOLUTION:
            theta_bounds = [(0.1, 4.0)]

            bound_list = theta_bounds[:]
            for _i in range(1, number_l + 1):
                bound_list.append((0.9, 4.0))

            bound = OptimizeMitigator(
                setting_h_mit=setting,
                print_x=print_x).diff_evolution(bound_list=bound_list)

        elif opt == OptMethod.BFGS:
            theta_start = 0.5

            start_list = [theta_start] + [1.0] * number_l

            bound = OptimizeMitigator(
                setting_h_mit=setting,
                print_x=print_x).bfgs(start_list=start_list)

        elif opt == OptMethod.GS_OLD:
            theta_bounds = [(0.1, 4.0)]

            bound_list = theta_bounds[:]
            for _i in range(1, number_l + 1):
                bound_list.append((0.9, 4.0))

            bound = OptimizeMitigator(
                setting_h_mit=setting, print_x=print_x).grid_search_old(
                    bound_list=bound_list, delta=0.1)

        elif opt == OptMethod.NM_OLD:
            nelder_mead_param = NelderMeadParameters()
            theta_start = 0.5

            start_list = [theta_start] + [1.0] * number_l
            start_simplex = InitialSimplex(
                parameters_to_optimize=number_l + 1).gao_han(
                    start_list=start_list)

            bound = OptimizeMitigator(
                setting_h_mit=setting, print_x=print_x).nelder_mead_old(
                    simplex=start_simplex,
                    nelder_mead_param=nelder_mead_param,
                    sd_min=10**(-2))

        else:
            raise NameError("Optimization parameter {0} is infeasible".format(
                opt.name))

        stop = timer()
        list_of_bounds.append(bound)
        list_of_times.append(stop - start)
        list_of_approaches.append(opt.name)

    print("list_of_approaches: ", list_of_approaches)
    print("list_of_times: ", list_of_times)
    print("list_of_bounds: ")
    return list_of_bounds


if __name__ == '__main__':
    from nc_operations.perform_enum import PerformEnum
    from nc_service.constant_rate_server import ConstantRate
    from nc_arrivals.qt import DM1
    from single_server.single_server_perform import SingleServerPerform
    from fat_tree.fat_cross_perform import FatCrossPerform
    from utils.perform_parameter import PerformParameter

    OUTPUT_TIME = PerformParameter(perform_metric=PerformEnum.OUTPUT, value=4)

    EXP_ARRIVAL = DM1(lamb=4.4)
    CONST_RATE = ConstantRate(rate=0.24)

    SETTING1 = SingleServerPerform(
        arr=EXP_ARRIVAL, const_rate=CONST_RATE, perform_param=OUTPUT_TIME)
    OPT_METHODS = [
        OptMethod.GRID_SEARCH, OptMethod.GS_OLD, OptMethod.PATTERN_SEARCH,
        OptMethod.BASIN_HOPPING, OptMethod.SIMULATED_ANNEALING,
        OptMethod.DIFFERENTIAL_EVOLUTION
    ]

    # print(
    #     compare_optimization(
    #         setting=SETTING1,
    #         opt_methods=OPT_METHODS,
    #         number_l=1))

    DELAY_PROB = PerformParameter(
        perform_metric=PerformEnum.DELAY_PROB, value=4)

    EXP_ARRIVAL1 = DM1(lamb=5.0)
    EXP_ARRIVAL2 = DM1(lamb=4.0)

    CONST_RATE1 = ConstantRate(rate=3.0)
    CONST_RATE2 = ConstantRate(rate=2.0)

    ARR_LIST = [EXP_ARRIVAL1, EXP_ARRIVAL2]
    SER_LIST = [CONST_RATE1, CONST_RATE2]

    SETTING2 = FatCrossPerform(
        arr_list=ARR_LIST, ser_list=SER_LIST, perform_param=DELAY_PROB)

    print(
        compare_optimization(
            setting=SETTING2, opt_methods=OPT_METHODS, number_l=1))
