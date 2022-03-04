"""Optimize theta"""

from math import inf
from typing import List

from nc_operations.e2e_enum import E2EEnum
from optimization.optimize import Optimize
from utils.exceptions import ParameterOutOfBounds
from utils.setting_pmoo import SettingPMOO


class OptimizePMOOBound(Optimize):
    """Optimize class"""
    def __init__(self, setting_pmoo: SettingPMOO, e2e_enum: E2EEnum,
                 number_param: int) -> None:
        super().__init__(setting=setting_pmoo, number_param=number_param)
        self.setting_pmoo = setting_pmoo
        self.e2e_enum = e2e_enum

    def eval_except(self, param_list: List[float]) -> float:
        """
        Shortens the exception handling and case distinction in a small method.

        :param param_list: theta parameter
        :return:           function to_value
        """
        try:
            if self.e2e_enum == E2EEnum.STANDARD:
                return self.setting_pmoo.standard_bound(param_list=param_list)

            elif self.e2e_enum == E2EEnum.ARR_RATE:
                return self.setting_pmoo.pmoo_arr_bound(param_list=param_list)

            elif self.e2e_enum == E2EEnum.MIN_RATE:
                return self.setting_pmoo.pmoo_min_bound(param_list=param_list)

            elif self.e2e_enum == E2EEnum.RATE_DIFF:
                return self.setting_pmoo.pmoo_rate_diff_bound(
                    param_list=param_list)

            elif self.e2e_enum == E2EEnum.ANALYTIC_COMBINATORICS:
                return self.setting_pmoo.pmoo_ac_bound(param_list=param_list)

            elif self.e2e_enum == E2EEnum.CUTTING:
                return self.setting_pmoo.cutting_bound(param_list=param_list)

            else:
                NotImplementedError("This analysis is not implemented")

        except (FloatingPointError, ParameterOutOfBounds, OverflowError,
                ZeroDivisionError):
            return inf
