"""We compare the different optimizations"""

from timeit import default_timer as timer
from typing import List

from h_mitigator.optimize_mitigator import OptimizeMitigator
from h_mitigator.setting_mitigator import SettingMitigator
from optimization.initial_simplex import InitialSimplex
from optimization.opt_method import OptMethod


def compare_optimization(setting: SettingMitigator,
                         opt_methods: List[OptMethod],
                         number_l=1) -> List[float]:
    """Measures time for different optimizations"""
    optim_mit = OptimizeMitigator(setting_h_mit=setting,
                                  number_param=number_l + 1)

    list_of_bounds: List[float] = []
    list_of_times: List[float] = []
    list_of_approaches: List[str] = []

    for opt in opt_methods:
        start = timer()

        match opt:
            case OptMethod.GRID_SEARCH:
                theta_bounds = [(0.1, 4.0)]

                bound_list = theta_bounds[:]
                for _i in range(number_l):
                    bound_list.append((0.9, 4.0))

                bound = optim_mit.grid_search(grid_bounds=bound_list,
                                              delta=0.1).obj_value

            case OptMethod.PATTERN_SEARCH:
                theta_start = 0.5

                start_list = [theta_start] + [1.0] * number_l

                bound = optim_mit.pattern_search(starting_point=start_list,
                                                 delta=3.0,
                                                 delta_min=0.01)

            case OptMethod.NELDER_MEAD:
                theta_start = 0.5

                start_list = [theta_start] + [1.0] * number_l
                start_simplex = InitialSimplex(parameters_to_optimize=number_l +
                                               1).gao_han(start_list=start_list)

                bound = optim_mit.nelder_mead(simplex=start_simplex,
                                              sd_min=10**(-2)).obj_value

            case OptMethod.BASIN_HOPPING:
                theta_start = 0.5

                start_list = [theta_start] + [1.0] * number_l

                bound = optim_mit.basin_hopping(
                    starting_point=start_list).obj_value

            case OptMethod.DUAL_ANNEALING:
                theta_bounds = [(0.1, 4.0)]

                bound_array = theta_bounds[:]
                for _i in range(1, number_l + 1):
                    bound_array.append((0.9, 4.0))

                bound = optim_mit.dual_annealing(
                    bound_list=bound_array).obj_value

            case OptMethod.DIFFERENTIAL_EVOLUTION:
                theta_bounds = [(0.1, 4.0)]

                bound_list = theta_bounds[:]
                for _i in range(number_l):
                    bound_list.append((0.9, 4.0))

                bound = optim_mit.diff_evolution(
                    bound_list=bound_list).obj_value

            case OptMethod.BFGS:
                theta_start = 0.5

                start_list = [theta_start] + [1.0] * number_l

                bound = optim_mit.bfgs(start_list=start_list).obj_value

            case _:
                raise NotImplementedError(f"Heuristic is not implemented")

        stop = timer()
        list_of_bounds.append(bound)
        list_of_times.append(stop - start)
        list_of_approaches.append(opt.name)

    print("list_of_approaches: ", list_of_approaches)
    print("list_of_runtimes: ", list_of_times)
    print("list_of_bounds: ")
    return list_of_bounds


if __name__ == '__main__':
    from nc_operations.perform_enum import PerformEnum
    from nc_server.constant_rate_server import ConstantRateServer
    from nc_arrivals.iid import DM1
    from h_mitigator.fat_cross_perform import FatCrossPerform
    from h_mitigator.single_server_mit_perform import SingleServerMitPerform
    from utils.perform_parameter import PerformParameter

    OUTPUT_TIME = PerformParameter(perform_metric=PerformEnum.OUTPUT, value=4)

    EXP_ARRIVAL = [DM1(lamb=4.4)]
    CONST_RATE = ConstantRateServer(rate=0.24)

    SETTING1 = SingleServerMitPerform(arr_list=EXP_ARRIVAL,
                                      server=CONST_RATE,
                                      perform_param=OUTPUT_TIME)
    OPT_METHODS = [
        OptMethod.GRID_SEARCH, OptMethod.GS_OLD, OptMethod.PATTERN_SEARCH,
        OptMethod.BASIN_HOPPING, OptMethod.DUAL_ANNEALING,
        OptMethod.DIFFERENTIAL_EVOLUTION
    ]

    # print(
    #     compare_optimization(
    #         setting=SETTING1,
    #         opt_methods=OPT_METHODS,
    #         number_l=1))

    DELAY_PROB = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                  value=4)

    EXP_ARRIVAL1 = DM1(lamb=5.0)
    EXP_ARRIVAL2 = DM1(lamb=4.0)

    CONST_RATE1 = ConstantRateServer(rate=3.0)
    CONST_RATE2 = ConstantRateServer(rate=2.0)

    ARR_LIST = [EXP_ARRIVAL1, EXP_ARRIVAL2]
    SER_LIST = [CONST_RATE1, CONST_RATE2]

    SETTING2 = FatCrossPerform(arr_list=ARR_LIST,
                               ser_list=SER_LIST,
                               perform_param=DELAY_PROB)

    print(
        compare_optimization(setting=SETTING2,
                             opt_methods=OPT_METHODS,
                             number_l=1))
