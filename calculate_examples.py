"""This serves as our main function."""

from fat_tree.fat_cross_perform import FatCrossPerform
from library.perform_parameter import PerformParameter
from nc_operations.perform_metric import PerformMetric
from nc_processes.arrival_distribution import MMOO, ExponentialArrival
from nc_processes.service import ConstantRate
from optimization.optimize import Optimize
from optimization.optimize_new import OptimizeNew
from optimization.simul_anneal_param import SimulAnnealParam
from single_server.single_server_perform import SingleServerPerform

if __name__ == '__main__':
    # Single server output calculation
    print("Single Server Performance Bounds:\n")

    OUTPUT_TIME6 = PerformParameter(
        perform_metric=PerformMetric.OUTPUT, value=6)

    SINGLE_SERVER = SingleServerPerform(
        arr=ExponentialArrival(lamb=1.0),
        ser=ConstantRate(rate=10.0),
        perform_param=OUTPUT_TIME6)

    print(SINGLE_SERVER.get_bound(0.1))

    print(SINGLE_SERVER.get_new_bound([0.1, 2.7]))

    print(
        Optimize(SINGLE_SERVER, print_x=True).grid_search(
            bound_list=[(0.1, 5.0)], delta=0.1))
    print(
        OptimizeNew(SINGLE_SERVER, print_x=True).grid_search(
            bound_list=[(0.1, 5.0), (0.9, 8.0)], delta=0.1))
    print(
        OptimizeNew(SINGLE_SERVER, print_x=True).pattern_search(
            start_list=[0.5, 1.0], delta=3, delta_min=0.01))

    OUTPUT_TIME2 = PerformParameter(
        perform_metric=PerformMetric.OUTPUT, value=2)

    SINGLE_SERVER2 = SingleServerPerform(
        arr=MMOO(mu=0.7, lamb=0.4, burst=1),
        ser=ConstantRate(rate=1),
        perform_param=OUTPUT_TIME2)
    print(
        Optimize(SINGLE_SERVER2, print_x=True).grid_search(
            bound_list=[(0.1, 5.0)], delta=0.1))
    print(
        OptimizeNew(SINGLE_SERVER2, print_x=True).grid_search(
            bound_list=[(0.1, 5.0), (0.9, 5.0)], delta=0.1))

    print("\n-------------------------------------------\n")

    # Fat cross delay probability calculation
    print("Fat Cross Performance Bounds:\n")

    DELAY_PROB6 = PerformParameter(
        perform_metric=PerformMetric.DELAY_PROB, value=6)

    EXAMPLE = FatCrossPerform(
        arr_list=[ExponentialArrival(lamb=1),
                  ExponentialArrival(lamb=4)],
        ser_list=[ConstantRate(rate=4),
                  ConstantRate(rate=0.5)],
        perform_param=DELAY_PROB6)
    print(EXAMPLE.get_bound(theta=0.3))
    print(EXAMPLE.get_new_bound(param_list=[0.3, 1.5]))

    DELAY_TIME = PerformParameter(
        perform_metric=PerformMetric.DELAY, value=0.032)

    EXAMPLE_REVERSE = FatCrossPerform(
        arr_list=[ExponentialArrival(lamb=1),
                  ExponentialArrival(lamb=4)],
        ser_list=[ConstantRate(rate=4),
                  ConstantRate(rate=0.5)],
        perform_param=DELAY_TIME)

    print(EXAMPLE_REVERSE.get_bound(theta=0.3))

    DELAY_PROB4 = PerformParameter(
        perform_metric=PerformMetric.DELAY_PROB, value=4)

    EXAMPLE2 = FatCrossPerform(
        arr_list=[ExponentialArrival(lamb=4),
                  ExponentialArrival(lamb=1)],
        ser_list=[ConstantRate(rate=3),
                  ConstantRate(rate=3)],
        perform_param=DELAY_PROB4)

    print(EXAMPLE2.get_bound(theta=0.3))
    print(EXAMPLE2.get_new_bound(param_list=[0.3, 3]))

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

    SIMU_ANNEAL_PARAM = SimulAnnealParam(
        rep_max=15, temp_start=1000.0, cooling_factor=0.95, search_radius=1.0)

    print(
        OPTIMIZE_NEW.simulated_annealing(
            start_list=[0.5, 1.0], simul_anneal_param=SIMU_ANNEAL_PARAM))
