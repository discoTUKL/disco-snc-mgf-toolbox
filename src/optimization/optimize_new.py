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

    def __init__(self,
                 setting_new: SettingNew,
                 new: bool = True,
                 print_x: bool = False,
                 show_warn: bool = False) -> None:
        super().__init__(setting_new, print_x, show_warn)
        self.setting_bound = setting_new
        self.new = new
        self.print_x = print_x
        self.show_warn = show_warn

    def eval_except(self, param_list: List[float]) -> float:
        """
        Shortens the exception handling and case distinction in a small method.

        :param param_list: theta parameter and Lyapunov parameters l_i
        :return:           function to_value
        """

        if self.new:
            try:
                return self.setting_bound.new_bound(param_l_list=param_list)
            except (ParameterOutOfBounds, OverflowError):
                return inf
        else:
            try:
                return self.setting_bound.bound(param_list=param_list)
            except (ParameterOutOfBounds, OverflowError):
                return inf


if __name__ == '__main__':
    from fat_tree.fat_cross_perform import FatCrossPerform
    from library.perform_parameter import PerformParameter
    from nc_operations.perform_enum import PerformEnum
    from nc_processes.arrival_distribution import MMOO
    from nc_processes.constant_rate_server import ConstantRate

    DELAY_4 = PerformParameter(perform_metric=PerformEnum.DELAY, value=0.0001)

    MMOO_1 = MMOO(mu=1.0, lamb=2.2, burst=3.4)
    MMOO_2 = MMOO(mu=3.6, lamb=1.6, burst=0.4)
    CONST_RATE_1 = ConstantRate(rate=2.0)
    CONST_RATE_2 = ConstantRate(rate=0.3)

    SIMPLEX_START = np.array([[0.1], [0.3]])
    # SIMPLEX_START = np.array([[100], [200]])
    SIMPLEX_START_NEW = np.array([[0.1, 2.0], [0.3, 1.2], [0.4, 1.1]])
    SIMPLEX_RAND = InitialSimplex(parameters_to_optimize=1).uniform_dist(
        max_theta=0.6, max_l=2.0)

    NM_PARAM_SET = NelderMeadParameters()

    SETTING = FatCrossPerform(
        arr_list=[MMOO_1, MMOO_2],
        ser_list=[CONST_RATE_1, CONST_RATE_2],
        perform_param=DELAY_4)

    OPTI_OLD = Optimize(setting=SETTING, print_x=True)
    print(OPTI_OLD.grid_search(bound_list=[(0.1, 4.0)], delta=0.1))
    print(OPTI_OLD.pattern_search(start_list=[0.5], delta=3.0, delta_min=0.01))
    print(Optimize.nelder_mead(self=OPTI_OLD, simplex=SIMPLEX_RAND))
    print(
        Optimize.nelder_mead_old(
            self=OPTI_OLD,
            simplex=SIMPLEX_RAND,
            nelder_mead_param=NM_PARAM_SET))
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
            simplex=SIMPLEX_START_NEW, nelder_mead_param=NM_PARAM_SET))
    print(OPTI_NEW.bfgs(start_list=[0.4] + [1.0]))
