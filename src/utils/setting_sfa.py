"""This superclass represents our get_value abstract class"""

from abc import abstractmethod
from typing import List

from utils.setting import Setting


class SettingSFA(Setting):
    """Each setting (topology) has to implements methods to obtain
    the bounds"""
    @abstractmethod
    def sfa_arr_bound(self, param_list: List[float]) -> float:
        """
        SFA analysis paying burst with the arrivals

        :param param_list: theta parameter
        """
        pass

    @abstractmethod
    def sfa_min_bound(self, param_list: List[float]) -> float:
        """
        SFA analysis paying burst with the min rate

        :param param_list: theta parameter
        """
        pass

    @abstractmethod
    def sfa_rate_diff_bound(self, param_list: List[float]) -> float:
        """
        SFA analysis paying burst with the min rate and rate-diff penalty

        :param param_list: theta parameter
        """
        pass

    @abstractmethod
    def sfa_ac_bound(self, param_list: List[float]) -> float:
        """
        SFA analysis using analytic combinatorics

        :param param_list: theta parameter
        """
        pass

    @abstractmethod
    def sfa_explicit(self, param_list: List[float]) -> float:
        """
        SFA explicit computation

        :param param_list: theta parameter
        """
        pass
