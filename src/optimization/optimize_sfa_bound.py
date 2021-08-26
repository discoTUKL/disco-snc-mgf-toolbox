"""Optimize theta"""

from math import inf
from typing import List

from nc_operations.e2e_enum import E2EEnum
from optimization.optimize import Optimize
from utils.exceptions import ParameterOutOfBounds
from utils.setting_sfa import SettingSFA


class OptimizeSFABound(Optimize):
    """Optimize class"""
    def __init__(self,
                 setting_sfa: SettingSFA,
                 e2e_enum: E2EEnum,
                 number_param: int,
                 print_x=False) -> None:
        super().__init__(setting=setting_sfa,
                         number_param=number_param,
                         print_x=print_x)
        self.setting_sfa = setting_sfa
        self.e2e_enum = e2e_enum
        self.number_param = number_param
        self.print_x = print_x

    def eval_except(self, param_list: List[float]) -> float:
        """
        Shortens the exception handling and case distinction in a small method.

        :param param_list: theta and Hoelder parameters
        :return:           function to_value
        """
        try:
            if self.e2e_enum == E2EEnum.STANDARD:
                return self.setting_sfa.standard_bound(param_list=param_list)

            elif self.e2e_enum == E2EEnum.ARR_RATE:
                return self.setting_sfa.sfa_arr_bound(param_list=param_list)

            elif self.e2e_enum == E2EEnum.MIN_RATE:
                return self.setting_sfa.sfa_min_bound(param_list=param_list)

            elif self.e2e_enum == E2EEnum.RATE_DIFF:
                return self.setting_sfa.sfa_rate_diff_bound(
                    param_list=param_list)

            elif self.e2e_enum == E2EEnum.ANALYTIC_COMBINATORICS:
                return self.setting_sfa.sfa_ac_bound(
                    param_list=param_list)

            else:
                NotImplementedError("This analysis is not implemented")

        except (FloatingPointError, ParameterOutOfBounds, OverflowError):
            return inf
