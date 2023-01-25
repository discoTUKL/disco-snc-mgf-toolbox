"""This superclass represents our get_value abstract class"""

from abc import abstractmethod
from typing import List

from utils.setting import Setting


class SettingPMOO(Setting):
    """Each setting (topology) has to implement methods to obtain
    the bounds"""
    @abstractmethod
    def pmoo_arr_bound(self, param_list: List[float]) -> float:
        """
        PMOO analysis paying burst with the arrivals

        :param param_list: theta parameter
        """
        pass

    @abstractmethod
    def pmoo_min_bound(self, param_list: List[float]) -> float:
        """
        PMOO analysis paying burst with the min rate

        :param param_list: theta parameter
        """
        pass

    @abstractmethod
    def pmoo_rate_diff_bound(self, param_list: List[float]) -> float:
        """
        PMOO analysis paying burst with the min rate and rate-diff penalty

        :param param_list: theta parameter
        """
        pass

    @abstractmethod
    def pmoo_ac_bound(self, param_list: List[float]) -> float:
        """
        PMOO analysis using analytic combinatorics

        :param param_list: theta parameter
        """
        pass

    def cutting_bound(self, param_list: List[float]) -> float:
        """
        Cutting technique between PMOO and SFA

        :param param_list: theta parameter
        """
        pass

    @abstractmethod
    def pmoo_binom_bound(self, param_list: List[float]) -> float:
        """
        PMOO analysis paying burst with the min rate and binomial distribution

        :param param_list: theta parameter
        """
        pass

    @abstractmethod
    def pmoo_new_bound(self, param_list: List[float]) -> float:
        """
        PMOO analysis with completely new bound

        :param param_list: theta parameter
        """
        pass

    @abstractmethod
    def pmoo_mixed_bound(self, param_list: List[float]) -> float:
        """
        PMOO mixed analysis

        :param param_list: theta parameter
        """
        pass

    @abstractmethod
    def pmoo_explicit(self, param_list: List[float]) -> float:
        """
        PMOO explicit computation

        :param param_list: theta parameter
        """
        pass
