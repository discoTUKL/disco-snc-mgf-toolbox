"""Small examples to play with."""

from typing import List

from h_mitigator.fat_cross_perform import FatCrossPerform
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.markov_modulated import MMOOCont
from nc_arrivals.iid import DM1
from nc_operations.single_server_perform import SingleServerPerform
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from optimization.optimize import Optimize
from optimization.sim_anneal_param import SimAnnealParams
from utils.perform_parameter import PerformParameter

if __name__ == '__main__':
    print("Single Server Performance Bounds:\n")

    DELAY_PROB8 = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                   value=8)

    SINGLE_SERVER = SingleServerPerform(foi=DM1(lamb=1.0),
                                        server=ConstantRateServer(rate=10.0),
                                        perform_param=DELAY_PROB8)

    print(
        Optimize(SINGLE_SERVER, number_param=1,
                 print_x=True).grid_search(grid_bounds=[(0.1, 5.0)], delta=0.1))

    SINGLE_SERVER2 = SingleServerPerform(
        foi=MMOOCont(mu=0.7, lamb=0.4, peak_rate=1.2),
        server=ConstantRateServer(rate=1.0),
        perform_param=DELAY_PROB8)

    print(
        Optimize(SINGLE_SERVER2, number_param=1,
                 print_x=True).grid_search(grid_bounds=[(0.1, 5.0)], delta=0.1))

    DELAY_PROB_REV = PerformParameter(perform_metric=PerformEnum.DELAY,
                                      value=0.0183)

    SINGLE_SERVER2 = SingleServerPerform(
        foi=MMOOCont(mu=0.7, lamb=0.4, peak_rate=1.2),
        server=ConstantRateServer(rate=1.0),
        perform_param=DELAY_PROB_REV)

    print(
        Optimize(SINGLE_SERVER2, number_param=1,
                 print_x=False).grid_search(grid_bounds=[(0.1, 5.0)],
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

    EXAMPLE_1 = FatCrossPerform(arr_list=ARR_LIST1,
                                ser_list=SER_LIST1,
                                perform_param=DELAY_PROB6)

    print("EXAMPLE_1\n")
    print(
        Optimize(setting=EXAMPLE_1, number_param=1,
                 print_x=True).grid_search(grid_bounds=[(0.1, 5.0)], delta=0.1))

    ARR_LIST2: List[ArrivalDistribution] = [
        MMOOCont(mu=0.5, lamb=0.5, peak_rate=1.5),
        MMOOCont(mu=0.5, lamb=0.5, peak_rate=0.7)
    ]

    SER_LIST2: List[ConstantRateServer] = [
        ConstantRateServer(rate=2.0),
        ConstantRateServer(rate=0.5)
    ]

    EXAMPLE_2 = FatCrossPerform(arr_list=ARR_LIST2,
                                ser_list=SER_LIST2,
                                perform_param=DELAY_PROB6)

    print("\nEXAMPLE_2\n")
    print(
        Optimize(EXAMPLE_2, number_param=1,
                 print_x=True).grid_search(grid_bounds=[(0.1, 5.0)], delta=0.1))

    DELAY_TIME = PerformParameter(perform_metric=PerformEnum.DELAY,
                                  value=0.010)

    EXAMPLE_REVERSE = FatCrossPerform(arr_list=ARR_LIST2,
                                      ser_list=SER_LIST2,
                                      perform_param=DELAY_TIME)

    print(
        Optimize(EXAMPLE_REVERSE, number_param=1,
                 print_x=True).grid_search(grid_bounds=[(0.1, 5.0)], delta=0.1))

    DELAY_PROB4 = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                   value=4)

    print(
        Optimize(EXAMPLE_2, number_param=1,
                 print_x=True).pattern_search(start_list=[0.5],
                                              delta=3,
                                              delta_min=0.01))

    SIMU_ANNEAL_PARAM = SimAnnealParams(rep_max=15,
                                        temp_start=1000.0,
                                        cooling_factor=0.95,
                                        search_radius=1.0)

    print(
        Optimize(EXAMPLE_2, number_param=1,
                 print_x=True).basin_hopping(start_list=[0.5]))

    # print(
    #     Optimize(EXAMPLE_2, number_param=1,
    #                       print_x=True).dual_annealing(bound_list=[(0.1, 5.0)]))

    print(
        Optimize(EXAMPLE_2, number_param=1, print_x=True).sim_annealing(
            start_list=[0.5], sim_anneal_params=SIMU_ANNEAL_PARAM))
