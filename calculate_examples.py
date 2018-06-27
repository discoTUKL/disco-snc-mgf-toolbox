"""This serves as our main function."""
# TODO: Write a test

from typing import List

from fat_tree.fat_cross_perform import FatCrossPerform
from library.perform_parameter import PerformParameter
from nc_operations.perform_metric import PerformMetric
from nc_processes.arrival_distribution import (MMOO, ArrivalDistribution,
                                               ExponentialArrival)
from nc_processes.constant_rate_server import ConstantRate
from optimization.optimize import Optimize
from optimization.optimize_new import OptimizeNew
from optimization.simul_annealing import SimulAnnealing
from single_server.single_server_perform import SingleServerPerform

if __name__ == '__main__':
    # Single server output calculation
    print("Single Server Performance Bounds:\n")

    OUTPUT_TIME6 = PerformParameter(
        perform_metric=PerformMetric.OUTPUT, value=6)

    SINGLE_SERVER = SingleServerPerform(
        arr=ExponentialArrival(lamb=1.0),
        const_rate=ConstantRate(rate=10.0),
        perform_param=OUTPUT_TIME6)

    print(SINGLE_SERVER.bound(0.1))

    print(SINGLE_SERVER.new_bound([0.1, 2.7]))

    print(
        Optimize(SINGLE_SERVER, print_x=True).grid_search(
            bound_list=[(0.1, 5.0)], delta=0.1))
    print(
        OptimizeNew(SINGLE_SERVER, print_x=True).grid_search(
            bound_list=[(0.1, 5.0), (0.9, 8.0)], delta=0.1))
    print(
        OptimizeNew(SINGLE_SERVER, print_x=True).pattern_search(
            start_list=[0.5, 1.0], delta=3, delta_min=0.01))

    DELAY_PROB10 = PerformParameter(
        perform_metric=PerformMetric.DELAY_PROB, value=10)

    SINGLE_SERVER2 = SingleServerPerform(
        arr=MMOO(mu=0.7, lamb=0.4, burst=1.2),
        const_rate=ConstantRate(rate=1),
        perform_param=DELAY_PROB10)

    print(
        Optimize(SINGLE_SERVER2, print_x=True).grid_search(
            bound_list=[(0.1, 5.0)], delta=0.1))

    # TODO: stops with error
    print(
        OptimizeNew(SINGLE_SERVER2, print_x=True).grid_search(
            bound_list=[(0.1, 5.0), (0.9, 8.0)], delta=0.1))

    print("\n-------------------------------------------\n")

    # Fat cross delay probability calculation
    print("Fat Cross Performance Bounds:\n")

    DELAY_PROB6 = PerformParameter(
        perform_metric=PerformMetric.DELAY_PROB, value=6)

    ARR_LIST: List[ArrivalDistribution] = [
        ExponentialArrival(lamb=1),
        ExponentialArrival(lamb=4)
    ]

    SER_LIST: List[ConstantRate] = [
        ConstantRate(rate=4), ConstantRate(rate=0.5)
    ]

    EXAMPLE = FatCrossPerform(
        arr_list=ARR_LIST, ser_list=SER_LIST, perform_param=DELAY_PROB6)
    print(EXAMPLE.bound(theta=0.3))
    print(EXAMPLE.new_bound(param_list=[0.3, 1.5]))

    DELAY_TIME = PerformParameter(
        perform_metric=PerformMetric.DELAY, value=0.032)

    EXAMPLE_REVERSE = FatCrossPerform(
        arr_list=ARR_LIST, ser_list=SER_LIST, perform_param=DELAY_TIME)

    print(EXAMPLE_REVERSE.bound(theta=0.3))

    DELAY_PROB10 = PerformParameter(
        perform_metric=PerformMetric.DELAY_PROB, value=4)

    SER_LIST2: List[ConstantRate] = [
        ConstantRate(rate=3), ConstantRate(rate=3)
    ]

    EXAMPLE2 = FatCrossPerform(
        arr_list=ARR_LIST, ser_list=SER_LIST2, perform_param=DELAY_PROB10)

    print(EXAMPLE2.bound(theta=0.3))
    print(EXAMPLE2.new_bound(param_list=[0.3, 3]))

    print(
        Optimize(EXAMPLE, print_x=True).grid_search_old(
            bound_list=[(0.1, 5.0)], delta=0.1))

    print(
        OptimizeNew(EXAMPLE, print_x=True).grid_search(
            bound_list=[(0.1, 5.0), (0.9, 5)], delta=0.1))

    print(
        Optimize(EXAMPLE, print_x=True).grid_search(
            bound_list=[(0.1, 5.0)], delta=0.1))

    OPTIMIZE_NEW = OptimizeNew(EXAMPLE, print_x=True)

    print(
        OPTIMIZE_NEW.pattern_search(
            start_list=[0.5, 1.0], delta=3, delta_min=0.01))

    SIMU_ANNEAL_PARAM = SimulAnnealing(
        rep_max=15, temp_start=1000.0, cooling_factor=0.95, search_radius=1.0)

    print(OPTIMIZE_NEW.basin_hopping(start_list=[0.5, 1.0]))

    print(
        OPTIMIZE_NEW.simulated_annealing(
            start_list=[0.5, 1.0], simul_annealing=SIMU_ANNEAL_PARAM))
