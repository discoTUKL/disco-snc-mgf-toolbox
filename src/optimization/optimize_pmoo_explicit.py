"""Optimize theta"""

from math import inf
from typing import List

from optimization.optimize import Optimize
from utils.exceptions import ParameterOutOfBounds
from utils.setting_pmoo import SettingPMOO


class OptimizePMOOExplicit(Optimize):
    """Optimize class"""
    def __init__(self, setting_pmoo: SettingPMOO, number_param: int) -> None:
        super().__init__(setting=setting_pmoo, number_param=number_param)
        self.setting_pmoo = setting_pmoo
        self.number_param = number_param

    def eval_except(self, param_list: List[float]) -> float:
        """
        Shortens the exception handling and case distinction in a small method.

        :param param_list: theta parameter
        :return:           function to_value
        """
        try:
            return self.setting_pmoo.pmoo_explicit(param_list=param_list)
        except (ParameterOutOfBounds, OverflowError, ZeroDivisionError):
            return inf
