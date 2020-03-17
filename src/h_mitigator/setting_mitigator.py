"""This superclass represents our get_value abstract class"""

from abc import abstractmethod
from typing import List

from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_server.server_distribution import ServerDistribution
from utils.perform_parameter import PerformParameter
from utils.setting import Setting


class SettingMitigator(Setting):
    """Each setting (topology) has to implements methods to obtain
    the bounds"""
    def __init__(self, arr_list: List[ArrivalDistribution],
                 ser_list: List[ServerDistribution],
                 perform_param: PerformParameter) -> None:
        super().__init__(arr_list, ser_list, perform_param)

    @abstractmethod
    def h_mit_bound(self, param_l_list: List[float]) -> float:
        """
        new Lyapunov standard_bound

        :param param_l_list: theta and Lyapunov parameters
        """
        pass
