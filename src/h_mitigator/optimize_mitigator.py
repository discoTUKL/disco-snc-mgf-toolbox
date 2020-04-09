"""Optimize theta and all Lyapunov l's"""

from math import inf
from typing import List

import numpy as np

from h_mitigator.setting_mitigator import SettingMitigator
from optimization.initial_simplex import InitialSimplex
from optimization.nelder_mead_parameters import NelderMeadParameters
from optimization.optimize import Optimize
from utils.exceptions import ParameterOutOfBounds


class OptimizeMitigator(Optimize):
    """Optimize class"""
    def __init__(self,
                 setting_h_mit: SettingMitigator,
                 number_param: int,
                 print_x=False) -> None:
        super().__init__(setting=setting_h_mit,
                         number_param=number_param,
                         print_x=print_x)
        self.setting_h_mit = setting_h_mit
        self.number_param = number_param
        self.print_x = print_x

    def eval_except(self, param_list: List[float]) -> float:
        """
        Shortens the exception handling and case distinction in a small method.

        :param param_list: theta parameter and Lyapunov parameters l_i
        :return:           function to_value
        """
        try:
            return self.setting_h_mit.h_mit_bound(param_l_list=param_list)
        except (ParameterOutOfBounds, OverflowError):
            return inf


if __name__ == '__main__':
    from h_mitigator.fat_cross_perform import FatCrossPerform
    from utils.perform_parameter import PerformParameter
    from nc_operations.perform_enum import PerformEnum
    from nc_server.constant_rate_server import ConstantRateServer
    from nc_arrivals.markov_modulated import MMOOFluid

    DELAY_4 = PerformParameter(perform_metric=PerformEnum.DELAY, value=0.0001)

    MMOO_1 = MMOOFluid(mu=1.0, lamb=2.2, peak_rate=3.4)
    MMOO_2 = MMOOFluid(mu=3.6, lamb=1.6, peak_rate=0.4)
    CONST_RATE_1 = ConstantRateServer(rate=2.0)
    CONST_RATE_2 = ConstantRateServer(rate=0.3)

    SIMPLEX_START = np.array([[0.1], [0.3]])
    # SIMPLEX_START = np.array([[100], [200]])
    SIMPLEX_START_NEW = np.array([[0.1, 2.0], [0.3, 1.2], [0.4, 1.1]])
    SIMPLEX_RAND = InitialSimplex(parameters_to_optimize=1).uniform_dist(
        max_theta=0.6, max_l=2.0)

    NM_PARAM_SET = NelderMeadParameters()

    SETTING = FatCrossPerform(arr_list=[MMOO_1, MMOO_2],
                              ser_list=[CONST_RATE_1, CONST_RATE_2],
                              perform_param=DELAY_4)

    OPTI_OLD = Optimize(setting=SETTING, number_param=1, print_x=True)
    print(OPTI_OLD.grid_search(bound_list=[(0.1, 4.0)], delta=0.1))
    print(OPTI_OLD.pattern_search(start_list=[0.5], delta=3.0, delta_min=0.01))
    print(Optimize.nelder_mead(self=OPTI_OLD, simplex=SIMPLEX_RAND))
    print(
        Optimize.nelder_mead_old(self=OPTI_OLD,
                                 simplex=SIMPLEX_RAND,
                                 nelder_mead_param=NM_PARAM_SET))
    print(OPTI_OLD.basin_hopping(start_list=[2.0]))
    print(OPTI_OLD.diff_evolution(bound_list=[(0.1, 4.0)]))
    print(OPTI_OLD.bfgs(start_list=[0.4]))

    OPTI_NEW = OptimizeMitigator(setting_h_mit=SETTING,
                                 number_param=2,
                                 print_x=True)
    print(
        OPTI_NEW.grid_search_old(bound_list=[(0.1, 4.0), (0.9, 4.0)],
                                 delta=0.1))
    print(OPTI_NEW.grid_search(bound_list=[(0.1, 4.0), (0.9, 4.0)], delta=0.1))
    print(
        OPTI_NEW.pattern_search(start_list=[0.5] + [1.0],
                                delta=3.0,
                                delta_min=0.01))
    print(OPTI_NEW.nelder_mead(simplex=SIMPLEX_START_NEW))
    print(
        OPTI_NEW.nelder_mead_old(simplex=SIMPLEX_START_NEW,
                                 nelder_mead_param=NM_PARAM_SET))
    print(OPTI_NEW.bfgs(start_list=[0.4] + [1.0]))
