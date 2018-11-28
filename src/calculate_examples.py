"""This serves as our main function."""
# TODO: Write a test

from typing import List

from fat_tree.fat_cross_perform import FatCrossPerform
from library.perform_parameter import PerformParameter
from nc_operations.perform_enum import PerformEnum
from nc_processes.arrival_distribution import ArrivalDistribution
from nc_processes.constant_rate_server import ConstantRate
from nc_processes.markov_modulated import MMOOFluid
from nc_processes.qt import DM1
from optimization.optimize import Optimize
from optimization.optimize_new import OptimizeNew
from single_server.single_server_perform import SingleServerPerform

if __name__ == '__main__':
    # Single server output calculation
    print("Single Server Performance Bounds:\n")

    OUTPUT_TIME6 = PerformParameter(perform_metric=PerformEnum.OUTPUT, value=6)

    SINGLE_SERVER = SingleServerPerform(
        arr=DM1(lamb=1.0),
        const_rate=ConstantRate(rate=10.0),
        perform_param=OUTPUT_TIME6)

    print(SINGLE_SERVER.bound(param_list=[0.1]))

    print(SINGLE_SERVER.new_bound(param_l_list=[0.1, 2.7]))

    print(
        Optimize(SINGLE_SERVER, print_x=True, show_warn=True).grid_search(
            bound_list=[(0.1, 5.0)], delta=0.1))
    print(
        OptimizeNew(SINGLE_SERVER, print_x=True, show_warn=True).grid_search(
            bound_list=[(0.1, 5.0), (0.9, 8.0)], delta=0.1))
    print(
        OptimizeNew(SINGLE_SERVER, print_x=True,
                    show_warn=True).pattern_search(
                        start_list=[0.5, 1.0], delta=3, delta_min=0.01))

    DELAY_PROB8 = PerformParameter(
        perform_metric=PerformEnum.DELAY_PROB, value=8)

    SINGLE_SERVER2 = SingleServerPerform(
        arr=MMOOFluid(mu=0.7, lamb=0.4, burst=1.2),
        const_rate=ConstantRate(rate=1.0),
        perform_param=DELAY_PROB8)

    print(
        Optimize(SINGLE_SERVER2, print_x=True, show_warn=True).grid_search_old(
            bound_list=[(0.1, 5.0)], delta=0.1))

    print(
        Optimize(SINGLE_SERVER2, print_x=True, show_warn=True).grid_search(
            bound_list=[(0.1, 5.0)], delta=0.1))

    print(
        OptimizeNew(SINGLE_SERVER2, print_x=True, show_warn=True).grid_search(
            bound_list=[(0.1, 5.0), (0.9, 6.0)], delta=0.1))

    print(
        OptimizeNew(SINGLE_SERVER2, print_x=True,
                    show_warn=True).diff_evolution(bound_list=[(0.1,
                                                                5.0), (0.9,
                                                                       6.0)]))

    DELAY_PROB_REV = PerformParameter(
        perform_metric=PerformEnum.DELAY, value=0.0183)

    SINGLE_SERVER2 = SingleServerPerform(
        arr=MMOOFluid(mu=0.7, lamb=0.4, burst=1.2),
        const_rate=ConstantRate(rate=1.0),
        perform_param=DELAY_PROB_REV)

    print(
        Optimize(SINGLE_SERVER2, show_warn=True).grid_search(
            bound_list=[(0.1, 5.0)], delta=0.1))

    print("\n-------------------------------------------\n")

    # Fat cross delay probability calculation
    print("Fat Cross Performance Bounds:\n")

    DELAY_PROB5 = PerformParameter(
        perform_metric=PerformEnum.DELAY_PROB, value=5)

    ARR_LIST: List[ArrivalDistribution] = [
        MMOOFluid(mu=0.5, lamb=0.5, burst=1.5),
        MMOOFluid(mu=0.5, lamb=0.5, burst=0.7)
    ]

    SER_LIST: List[ConstantRate] = [
        ConstantRate(rate=2.5), ConstantRate(rate=0.5)
    ]

    EXAMPLE = FatCrossPerform(
        arr_list=ARR_LIST, ser_list=SER_LIST, perform_param=DELAY_PROB5)

    print(
        Optimize(EXAMPLE, print_x=True, show_warn=True).grid_search(
            bound_list=[(0.1, 5.0)], delta=0.1))

    DELAY_TIME = PerformParameter(
        perform_metric=PerformEnum.DELAY, value=0.0013)

    EXAMPLE_REVERSE = FatCrossPerform(
        arr_list=ARR_LIST, ser_list=SER_LIST, perform_param=DELAY_TIME)

    print(
        Optimize(EXAMPLE_REVERSE, print_x=True, show_warn=True).grid_search(
            bound_list=[(0.1, 5.0)], delta=0.1))

    # TODO: Reverse does not work for continuous time and multiplie servers

    # DELAY_PROB4 = PerformParameter(
    #     perform_metric=PerformEnum.DELAY_PROB, value=4)
    #
    # print(
    #     Optimize(EXAMPLE, print_x=True, show_warn=True).grid_search_old(
    #         bound_list=[(0.1, 5.0)], delta=0.1))
    #
    # print(
    #     OptimizeNew(EXAMPLE, print_x=True, show_warn=True).grid_search(
    #         bound_list=[(0.1, 5.0), (0.9, 5)], delta=0.1))
    #
    # print(
    #     Optimize(EXAMPLE, print_x=True, show_warn=True).grid_search(
    #         bound_list=[(0.1, 5.0)], delta=0.1))

    # OPTIMIZE_NEW = OptimizeNew(EXAMPLE, print_x=True, show_warn=True)

    # print(
    #     OPTIMIZE_NEW.pattern_search(
    #         start_list=[0.5, 1.0], delta=3, delta_min=0.01))
    #
    # SIMU_ANNEAL_PARAM = SimulAnnealing(
    #     rep_max=15, temp_start=1000.0, cooling_factor=0.95, search_radius=1.0)
    #
    # print(OPTIMIZE_NEW.basin_hopping(start_list=[0.5, 1.0]))
    #
    # print(
    #     OPTIMIZE_NEW.sim_annealing(
    #         start_list=[0.5, 1.0], simul_annealing=SIMU_ANNEAL_PARAM))
