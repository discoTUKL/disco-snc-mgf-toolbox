"""Optimize theta"""

from math import inf
from typing import List

from optimization.optimize import Optimize
from utils.exceptions import ParameterOutOfBounds

from msob_and_fp.setting_avoid_dep import SettingMSOBFP


class OptimizeFPBound(Optimize):
    """Optimize class"""
    def __init__(self,
                 setting_msob_fp: SettingMSOBFP,
                 number_param: int,
                 print_x=False) -> None:
        super().__init__(setting=setting_msob_fp,
                         number_param=number_param,
                         print_x=print_x)
        self.setting_msob_fp = setting_msob_fp
        self.number_param = number_param
        self.print_x = print_x

    def eval_except(self, param_list: List[float]) -> float:
        """
        Shortens the exception handling and case distinction in a small method.

        :param param_list: theta parameter
        :return:           function to_value
        """
        try:
            return self.setting_msob_fp.fp_bound(param_list=param_list)
        except (ParameterOutOfBounds, OverflowError):
            return inf
