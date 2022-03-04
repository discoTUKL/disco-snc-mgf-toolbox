"""Optimize theta"""

from math import inf
from typing import List

from optimization.optimize import Optimize
from utils.exceptions import ParameterOutOfBounds
from utils.setting_sfa import SettingSFA


class OptimizeSFAExplicit(Optimize):
    """Optimize class"""
    def __init__(self, setting_sfa: SettingSFA, number_param: int) -> None:
        super().__init__(setting=setting_sfa, number_param=number_param)
        self.setting_sfa = setting_sfa
        self.number_param = number_param

    def eval_except(self, param_list: List[float]) -> float:
        """
        Shortens the exception handling and case distinction in a small method.

        :param param_list: theta parameter
        :return:           function to_value
        """
        try:
            return self.setting_sfa.sfa_explicit(param_list=param_list)
        except (ParameterOutOfBounds, OverflowError, ZeroDivisionError):
            return inf
