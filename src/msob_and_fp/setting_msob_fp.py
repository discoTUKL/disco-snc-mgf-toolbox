"""This superclass represents our get_value abstract class"""

from abc import abstractmethod
from typing import List

from utils.setting import Setting


class SettingMSOBFP(Setting):
    @abstractmethod
    def server_bound(self, param_list: List[float]) -> float:
        """
        new bound using the server standard_bound

        :param param_list: theta parameter
        """
        pass

    @abstractmethod
    def fp_bound(self, param_list: List[float]) -> float:
        """
        new bound using the flow prolongation

        :param param_list: theta parameter
        """
        pass

    @abstractmethod
    def server_util(self, server_index: int) -> float:
        """
        :param server_index: index of one particular server
        :return: utilization of this server
        """
        pass
