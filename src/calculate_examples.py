"""Small examples to play with."""
# TODO: Add .to_name-method to classes

from typing import List

from h_mitigator.fat_cross_perform import FatCrossPerform
from h_mitigator.optimize_mitigator import OptimizeMitigator
from h_mitigator.single_server_mit_perform import SingleServerMitPerform
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.markov_modulated import MMOOFluid
from nc_arrivals.qt import DM1
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.optimize import Optimize
from utils.perform_parameter import PerformParameter

if __name__ == '__main__':
    print("Single Server Performance Bounds:\n")

    OUTPUT_TIME6 = PerformParameter(perform_metric=PerformEnum.OUTPUT, value=6)

    SINGLE_SERVER = SingleServerMitPerform(
        arr_list=[DM1(lamb=1.0)],
        ser_list=[ConstantRateServer(rate=10.0)],
        perform_param=OUTPUT_TIME6)

    print(SINGLE_SERVER.standard_bound(param_list=[0.1]))

    print(SINGLE_SERVER.h_mit_bound(param_l_list=[0.1, 2.7]))

    print(
        Optimize(SINGLE_SERVER, number_param=1,
                 print_x=True).grid_search(bound_list=[(0.1, 5.0)], delta=0.1))
    print(
        OptimizeMitigator(SINGLE_SERVER, number_param=2,
                          print_x=True).grid_search(bound_list=[(0.1, 5.0),
                                                                (0.9, 8.0)],
                                                    delta=0.1))
    print(
        OptimizeMitigator(SINGLE_SERVER, number_param=2,
                          print_x=True).pattern_search(start_list=[0.5, 1.0],
                                                       delta=3,
                                                       delta_min=0.01))

    DELAY_PROB8 = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                   value=8)

    SINGLE_SERVER2 = SingleServerMitPerform(
        arr_list=[MMOOFluid(mu=0.7, lamb=0.4, burst=1.2)],
        ser_list=[ConstantRateServer(rate=1.0)],
        perform_param=DELAY_PROB8)

    print(
        Optimize(SINGLE_SERVER2, number_param=1,
                 print_x=True).grid_search_old(bound_list=[(0.1, 5.0)],
                                               delta=0.1))

    print(
        Optimize(SINGLE_SERVER2, number_param=1,
                 print_x=True).grid_search(bound_list=[(0.1, 5.0)], delta=0.1))

    print(
        OptimizeMitigator(SINGLE_SERVER2, number_param=2,
                          print_x=True).grid_search(bound_list=[(0.1, 5.0),
                                                                (0.9, 6.0)],
                                                    delta=0.1))

    print(
        OptimizeMitigator(
            SINGLE_SERVER2, number_param=2,
            print_x=True).diff_evolution(bound_list=[(0.1, 5.0), (0.9, 6.0)]))

    DELAY_PROB_REV = PerformParameter(perform_metric=PerformEnum.DELAY,
                                      value=0.0183)

    SINGLE_SERVER2 = SingleServerMitPerform(
        arr_list=[MMOOFluid(mu=0.7, lamb=0.4, burst=1.2)],
        ser_list=[ConstantRateServer(rate=1.0)],
        perform_param=DELAY_PROB_REV)

    print(
        Optimize(SINGLE_SERVER2, number_param=1,
                 print_x=False).grid_search(bound_list=[(0.1, 5.0)],
                                            delta=0.1))

    print("\n-------------------------------------------\n")
    print("Fat Cross Performance Bounds:\n")

    DELAY_PROB6 = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                   value=6)

    ARR_LIST1: List[ArrivalDistribution] = [DM1(lamb=1.0), DM1(lamb=4.0)]

    SER_LIST1: List[ConstantRateServer] = [
        ConstantRateServer(rate=4.0),
        ConstantRateServer(rate=0.5)
    ]

    print(
        FatCrossPerform(arr_list=ARR_LIST1,
                        ser_list=SER_LIST1,
                        perform_param=DELAY_PROB6).standard_bound([0.3]))

    ARR_LIST2: List[ArrivalDistribution] = [
        MMOOFluid(mu=0.5, lamb=0.5, burst=1.5),
        MMOOFluid(mu=0.5, lamb=0.5, burst=0.7)
    ]

    SER_LIST2: List[ConstantRateServer] = [
        ConstantRateServer(rate=2.0),
        ConstantRateServer(rate=0.5)
    ]

    EXAMPLE = FatCrossPerform(arr_list=ARR_LIST2,
                              ser_list=SER_LIST2,
                              perform_param=DELAY_PROB6)

    print(
        Optimize(EXAMPLE, number_param=1,
                 print_x=True).grid_search(bound_list=[(0.1, 5.0)], delta=0.1))

    DELAY_TIME = PerformParameter(perform_metric=PerformEnum.DELAY,
                                  value=0.010)

    EXAMPLE_REVERSE = FatCrossPerform(arr_list=ARR_LIST2,
                                      ser_list=SER_LIST2,
                                      perform_param=DELAY_TIME)

    print(
        Optimize(EXAMPLE_REVERSE, number_param=1,
                 print_x=True).grid_search(bound_list=[(0.1, 5.0)], delta=0.1))

    # DELAY_PROB4 = PerformParameter(
    #     perform_metric=PerformEnum.DELAY_PROB, value=4)
    #
    # print(
    #     Optimize(EXAMPLE, number_parameters=1, print_x=True).grid_search_old(
    #         bound_list=[(0.1, 5.0)], delta=0.1))
    #
    # print(
    #     OptimizeMitigator(EXAMPLE,
    #                       number_parameters=2,
    #                       print_x=True).grid_search(
    #         bound_list=[(0.1, 5.0), (0.9, 5)], delta=0.1))
    #
    # print(
    #     Optimize(EXAMPLE, print_x=True, number_parameters=1).grid_search(
    #         bound_list=[(0.1, 5.0)], delta=0.1))
    #
    # OPTIMIZE_NEW = OptimizeMitigator(EXAMPLE,
    #                                  number_parameters=2,
    #                                  print_x=True)
    #
    # print(
    #     OPTIMIZE_NEW.pattern_search(
    #         start_list=[0.5, 1.0], delta=3, delta_min=0.01))
    #
    # SIMU_ANNEAL_PARAM = SimAnnealParams(
    #     rep_max=15, temp_start=1000.0, cooling_factor=0.95, search_radius=1.0)
    #
    # print(OPTIMIZE_NEW.basin_hopping(start_list=[0.5, 1.0]))
    #
    # print(
    #     OPTIMIZE_NEW.sim_annealing(
    #         start_list=[0.5, 1.0], sim_anneal_params=SIMU_ANNEAL_PARAM))
