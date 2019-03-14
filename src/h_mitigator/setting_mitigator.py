"""This superclass represents our get_value abstract class"""

from abc import abstractmethod
from typing import List

from utils.setting import Setting


class SettingMitigator(Setting):
    """Each setting (topology) has to implements methods to obtain
    the bounds"""

    @abstractmethod
    def h_mit_bound(self, param_l_list: List[float]) -> float:
        """
        new Lyapunov bound

        :param param_l_list: theta and Lyapunov parameters
        """
        pass
