"""This superclass represents our get_value abstract class"""

from abc import abstractmethod
from typing import List

from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_server.server_distribution import ServerDistribution
from utils.perform_parameter import PerformParameter


class Setting(object):
    """Each setting (topology) has to implements methods to obtain
    the bounds"""
    def __init__(self, arr_list: List[ArrivalDistribution],
                 ser_list: List[ServerDistribution],
                 perform_param: PerformParameter) -> None:
        self.arr_list = arr_list
        self.ser_list = ser_list
        self.perform_param = perform_param

    @abstractmethod
    def standard_bound(self, param_list: List[float]) -> float:
        """
        standard bound

        :param param_list: theta and Hoelder parameters
        """
        pass

    @abstractmethod
    def approximate_utilization(self) -> float:
        pass

    def to_name(self) -> str:
        return self.__class__.__name__
