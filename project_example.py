from fat_tree.fat_cross_perform import FatCrossPerform
from optimization.optimize import Optimize
from library.perform_parameter import PerformParameter
from nc_processes.arrival_distribution import ExponentialArrival
from nc_operations.perform_metric import PerformMetric
from nc_processes.constant_rate_server import ConstantRate

if __name__ == '__main__':
    PROB_VALUES = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0]

    for p in PROB_VALUES:
        DELAY_TIME = PerformParameter(
            perform_metric=PerformMetric.DELAY, value=p)

        EXAMPLE = FatCrossPerform(
            arr_list=[ExponentialArrival(lamb=1)],
            ser_list=[ConstantRate(rate=2)],
            perform_param=DELAY_TIME)

        print(
            Optimize(setting=EXAMPLE, print_x=False).grid_search(
                bound_list=[(0.01, 1.1)], delta=0.01))
