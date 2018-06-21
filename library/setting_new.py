"""This superclass represents our get_value abstract class"""

from abc import abstractmethod
from library.setting import Setting
from typing import List


class SettingNew(Setting):
    """Each setting (topology) has to implements methods to obtain
    the bounds"""

    @abstractmethod
    def bound(self, theta: float) -> float:
        """
        standard bound

        :param theta: mgf parameter
        """
        pass

    @abstractmethod
    def new_bound(self, param_list: List[float]) -> float:
        """
        new Lyapunov bound

        :param param_list: theta and Lyapunov parameters
        """
        pass
