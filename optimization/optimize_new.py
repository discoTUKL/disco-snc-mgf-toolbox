"""Optimize theta and all Lyapunov l's"""

from math import inf
from typing import List

import numpy as np

from library.exceptions import ParameterOutOfBounds
from library.setting_new import SettingNew
from optimization.initial_simplex import InitialSimplex
from optimization.nelder_mead_parameters import NelderMeadParameters
from optimization.optimize import Optimize


class OptimizeNew(Optimize):
    """Optimize class"""

    def __init__(self, setting_new: SettingNew, new=True,
                 print_x=False) -> None:
        super().__init__(setting_new, print_x)
        self.setting_bound = setting_new
        self.new = new
        self.print_x = print_x

    def eval_except(self, param_list: List[float]) -> float:
        """
        Shortens the exception handling and case distinction in a small method.

        :param param_list: theta parameter and Lyapunov parameters l_i
        :return:           function value
        """
        # debug code:

        # if self.new:
        #     res = self.setting.new_bound(param_list=param_list)
        #
        # else:
        #     res = self.setting.bound(theta=param_list[0])

        if self.new:
            try:
                res = self.setting_bound.new_bound(param_list=param_list)
            except (ParameterOutOfBounds, OverflowError):
                res = inf
        else:
            try:
                res = self.setting_bound.bound(theta=param_list[0])
            except (ParameterOutOfBounds, OverflowError):
                res = inf

        return res


if __name__ == '__main__':
    from nc_operations.perform_metric import PerformMetric
    from nc_processes.arrival_distribution import MMOO
    from nc_processes.service_distribution import ConstantRate
    from fat_tree.fat_cross_perform import FatCrossPerform
    from library.perform_parameter import PerformParameter

    DELAY_4 = PerformParameter(
        perform_metric=PerformMetric.DELAY, value=0.0001)

    mmoo1 = MMOO(mu=1.0, lamb=2.2, burst=3.4)
    mmoo2 = MMOO(mu=3.6, lamb=1.6, burst=0.4)
    const_rate1 = ConstantRate(rate=2.0)
    const_rate2 = ConstantRate(rate=0.3)

    SIMPLEX_START = np.array([[0.1], [0.3]])
    # SIMPLEX_START = np.array([[100], [200]])
    SIMPLEX_START_NEW = np.array([[0.1, 2.0], [0.3, 1.2], [0.4, 1.1]])
    SIMPLEX_RAND = InitialSimplex(parameters_to_optimize=1).uniform_dist(
        max_theta=0.6, max_l=2.0)

    nelder_mead_param_set = NelderMeadParameters()

    SETTING = FatCrossPerform(
        arr_list=[mmoo1, mmoo2],
        ser_list=[const_rate1, const_rate2],
        perform_param=DELAY_4)

    OPTI_OLD = Optimize(setting=SETTING, print_x=True)
    print(OPTI_OLD.grid_search(bound_list=[(0.1, 4.0)], delta=0.1))
    print(OPTI_OLD.pattern_search(start_list=[0.5], delta=3.0, delta_min=0.01))
    print(Optimize.nelder_mead(self=OPTI_OLD, simplex=SIMPLEX_RAND))
    print(
        Optimize.nelder_mead_old(
            self=OPTI_OLD,
            simplex=SIMPLEX_RAND,
            nelder_mead_param=nelder_mead_param_set))
    print(OPTI_OLD.basin_hopping(start_list=[2.0]))
    print(OPTI_OLD.differential_evolution(bound_list=[(0.1, 4.0)]))
    print(OPTI_OLD.bfgs(start_list=[0.4]))

    OPTI_NEW = OptimizeNew(setting_new=SETTING, new=True, print_x=True)
    print(
        OPTI_NEW.grid_search_old(
            bound_list=[(0.1, 4.0), (0.9, 4.0)], delta=0.1))
    print(OPTI_NEW.grid_search(bound_list=[(0.1, 4.0), (0.9, 4.0)], delta=0.1))
    print(
        OPTI_NEW.pattern_search(
            start_list=[0.5] + [1.0], delta=3.0, delta_min=0.01))
    print(OPTI_NEW.nelder_mead(simplex=SIMPLEX_START_NEW))
    print(
        OPTI_NEW.nelder_mead_old(
            simplex=SIMPLEX_START_NEW,
            nelder_mead_param=nelder_mead_param_set))
    print(OPTI_NEW.bfgs(start_list=[0.4] + [1.0]))
