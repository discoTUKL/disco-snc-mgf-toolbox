"""This superclass represents our get_value abstract class"""

from abc import ABC, abstractmethod
from typing import List


class Setting(ABC):
    """Each setting (topology) has to implements methods to obtain
    the bounds"""

    @abstractmethod
    def bound(self, param_list: List[float]) -> float:
        """
        standard bound

        :param param_list: theta and Hoelder parameters
        """
        pass

    def to_name(self) -> str:
        return self.__class__.to_name()
