"""Compare standard bound with Lyapunov bound."""

from math import nan
from timeit import default_timer as timer

from library.setting_new import SettingNew
from nc_operations.perform_metric import PerformMetric
from optimization.initial_simplex import InitialSimplex
from optimization.nelder_mead_parameters import NelderMeadParameters
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize
from optimization.optimize_new import OptimizeNew
from optimization.simul_anneal_param import SimulAnnealParam


def compute_improvement(setting: SettingNew,
                        opt_method: OptMethod,
                        number_l=1,
                        print_x=False) -> tuple:
    """Compare standard_bound with the new Lyapunov bound."""

    if opt_method == OptMethod.GRID_SEARCH:
        theta_bounds = [(0.1, 4.0)]

        bound_array = theta_bounds[:]
        # standard_bound_depr = Optimize(
        #     setting=setting, print_x=print_x).grid_search_old(
        #         bound_list=bound_array, delta=0.1)
        standard_bound = Optimize(
            setting=setting, print_x=print_x).grid_search(
                bound_list=bound_array, delta=0.1)

        bound_array_new = theta_bounds[:]
        for _i in range(1, number_l + 1):
            bound_array_new.append((0.9, 4.0))

        # new_bound_depr = OptimizeNew(
        #     setting_new=setting, new=True, print_x=print_x).grid_search_old(
        #         bound_list=bound_array_new, delta=0.1)
        new_bound = OptimizeNew(
            setting_new=setting, print_x=print_x).grid_search(
                bound_list=bound_array_new, delta=0.1)

        # print("Deprecated GS: ", standard_bound_depr, new_bound_depr)

        # if new_bound > standard_bound:
        #     print(
        #         Optimize(setting=setting,
        #                  print_x=True).grid_search_old(
        #                  bound_list=bound_array, delta=0.1))
        #     print(
        #         OptimizeNew(setting=setting,
        #                     print_x=True).grid_search_old(
        #                     bound_list=bound_array_new, delta=0.1))

    elif opt_method == OptMethod.PATTERN_SEARCH:
        theta_start = 0.5

        start_list = [theta_start]
        standard_bound = Optimize(
            setting=setting, print_x=print_x).pattern_search(
                start_list=start_list, delta=3.0, delta_min=0.01)

        start_list_new = [theta_start] + [1.0] * number_l

        new_bound = OptimizeNew(
            setting_new=setting, print_x=print_x).pattern_search(
                start_list=start_list_new, delta=3.0, delta_min=0.01)

        # This part is there to overcome opt_method issues
        if new_bound > standard_bound:
            new_bound = standard_bound

        # if new_bound == 0:
        #     print(Optimize(setting=setting,
        #                    print_x=True).pattern_search(
        #         start_dict={"theta": 0.5, "l_1": 1}, delta=3, delta_min=0.01
        #     ))
        #     print(OptimizeNew(setting=setting,
        #                    print_x=True).pattern_search(
        #         start_dict={"theta": 0.5, "l_1": 1}, delta=3, delta_min=0.01
        #     ))

    elif opt_method == OptMethod.NELDER_MEAD:
        nelder_mead_param = NelderMeadParameters()
        theta_start = 0.5

        start_list = [theta_start]
        start_simplex = InitialSimplex(parameters_to_optimize=1).gao_han(
            start_list=start_list)

        standard_bound_depr = Optimize(
            setting=setting, print_x=print_x).nelder_mead_old(
                simplex=start_simplex,
                nelder_mead_param=nelder_mead_param,
                sd_min=10**(-2))
        standard_bound = Optimize(
            setting=setting, print_x=print_x).nelder_mead(
                simplex=start_simplex, sd_min=10**(-2))

        start_list_new = [theta_start] + [1.0] * number_l
        start_simplex_new = InitialSimplex(parameters_to_optimize=number_l + 1
                                           ).gao_han(start_list=start_list_new)

        new_bound_depr = OptimizeNew(
            setting_new=setting, print_x=print_x).nelder_mead_old(
                simplex=start_simplex_new,
                nelder_mead_param=nelder_mead_param,
                sd_min=10**(-2))
        new_bound = OptimizeNew(
            setting_new=setting, print_x=print_x).nelder_mead(
                simplex=start_simplex_new, sd_min=10**(-2))

        # This part is there to overcome opt_method issues
        if new_bound > standard_bound:
            new_bound = standard_bound

        print("Deprecated NM: ", standard_bound_depr, new_bound_depr)

    elif opt_method == OptMethod.SIMULATED_ANNEALING:
        simul_anneal_param = SimulAnnealParam()
        theta_start = 0.5

        start_list = [theta_start]

        standard_bound = Optimize(
            setting=setting, print_x=print_x).simulated_annealing(
                start_list=start_list, simul_anneal_param=simul_anneal_param)

        start_list_new = [theta_start] + [1.0] * number_l

        new_bound = OptimizeNew(
            setting_new=setting, print_x=print_x).simulated_annealing(
                start_list=start_list_new,
                simul_anneal_param=simul_anneal_param)

        # This part is there to overcome opt_method issues
        if new_bound > standard_bound:
            new_bound = standard_bound

    else:
        raise NameError("Optimization parameter {0} is infeasible".format(
            opt_method.name))

    if standard_bound == 0 or new_bound == 0:
        standard_bound = nan
        new_bound = nan

    return standard_bound, new_bound


def compute_overhead(setting: SettingNew, opt_method: OptMethod,
                     number_l=1) -> tuple:
    """Compare computation times."""

    if opt_method == OptMethod.GRID_SEARCH:
        bound_array = [(0.1, 4.0)]

        start = timer()
        Optimize(setting=setting).grid_search_old(
            bound_list=bound_array, delta=0.1)
        stop = timer()
        time_standard_depr = stop - start

        start = timer()
        Optimize(setting=setting).grid_search(
            bound_list=bound_array, delta=0.1)
        stop = timer()
        time_standard = stop - start

        for _ in range(1, number_l + 1):
            bound_array.append((0.9, 4.0))

        start = timer()
        OptimizeNew(setting_new=setting).grid_search_old(
            bound_list=bound_array, delta=0.1)
        stop = timer()

        time_lyapunov_depr = stop - start
        start = timer()
        OptimizeNew(setting_new=setting).grid_search(
            bound_list=bound_array, delta=0.1)
        stop = timer()
        time_lyapunov = stop - start

        print("Deprecated GS: ", time_standard_depr, time_lyapunov_depr)

    elif opt_method == OptMethod.PATTERN_SEARCH:
        start_list = [0.5]

        start = timer()
        Optimize(setting=setting).pattern_search(
            start_list=start_list, delta=3.0, delta_min=0.01)
        stop = timer()
        time_standard = stop - start

        start_list = [0.5] + [1.0] * number_l

        start = timer()
        OptimizeNew(setting_new=setting).pattern_search(
            start_list=start_list, delta=3.0, delta_min=0.01)
        stop = timer()
        time_lyapunov = stop - start

    elif opt_method == OptMethod.NELDER_MEAD:
        nelder_mead_param = NelderMeadParameters()

        start_simplex = InitialSimplex(parameters_to_optimize=1).uniform_dist(
            max_theta=1.0)

        start = timer()
        Optimize(setting=setting).nelder_mead_old(
            simplex=start_simplex,
            nelder_mead_param=nelder_mead_param,
            sd_min=10**(-2))
        stop = timer()
        time_standard_depr = stop - start

        start = timer()
        Optimize(setting=setting).nelder_mead(
            simplex=start_simplex, sd_min=10**(-2))
        stop = timer()
        time_standard = stop - start

        start_simplex_new = InitialSimplex(parameters_to_optimize=number_l +
                                           1).uniform_dist(
                                               max_theta=1.0, max_l=2.0)

        start = timer()
        OptimizeNew(setting_new=setting).nelder_mead_old(
            simplex=start_simplex_new,
            nelder_mead_param=nelder_mead_param,
            sd_min=10**(-2))
        stop = timer()
        time_lyapunov_depr = stop - start

        start = timer()
        OptimizeNew(setting_new=setting).nelder_mead(
            simplex=start_simplex_new, sd_min=10**(-2))
        stop = timer()
        time_lyapunov = stop - start

        print("Deprecated NM: ", time_standard_depr, time_lyapunov_depr)

    else:
        raise NameError("Optimization parameter {0} is infeasible".format(
            opt_method.name))

    return time_standard, time_lyapunov


if __name__ == '__main__':
    from nc_processes.arrival_distribution import ExponentialArrival
    from nc_processes.service_distribution import ConstantRate
    from single_server.single_server_perform import SingleServerPerform
    from fat_tree.fat_cross_perform import FatCrossPerform
    from library.perform_parameter import PerformParameter

    OUTPUT_TIME = PerformParameter(
        perform_metric=PerformMetric.OUTPUT, value=4)

    EXP_ARRIVAL = ExponentialArrival(lamb=4.4)
    CONST_RATE = ConstantRate(rate=0.24)

    SETTING1 = SingleServerPerform(
        arr=EXP_ARRIVAL, ser=CONST_RATE, perform_param=OUTPUT_TIME)

    # print(
    #     compute_improvement(
    #         setting=SETTING1, opt_method=OptMethod.GRID_SEARCH,
    #         print_x=True))

    DELAY_PROB = PerformParameter(
        perform_metric=PerformMetric.DELAY_PROB, value=4)

    EXP_ARRIVAL1 = ExponentialArrival(lamb=11.0)
    EXP_ARRIVAL2 = ExponentialArrival(lamb=9.0)

    CONST_RATE1 = ConstantRate(rate=5.0)
    CONST_RATE2 = ConstantRate(rate=4.0)

    ARR_LIST = [EXP_ARRIVAL1, EXP_ARRIVAL2]
    SER_LIST = [CONST_RATE1, CONST_RATE2]

    SETTING2 = FatCrossPerform(
        arr_list=ARR_LIST, ser_list=SER_LIST, perform_param=DELAY_PROB)

    # print(
    #     compute_improvement(
    #         setting=SETTING2, opt_method=OptMethod.GRID_SEARCH,
    #         print_x=True))
    #
    # print(
    #     compute_improvement(
    #         setting=SETTING2,
    #         opt_method=OptMethod.PATTERN_SEARCH,
    #         print_x=True))

    print(
        compute_overhead(
            setting=SETTING2, opt_method=OptMethod.GRID_SEARCH, number_l=1))
