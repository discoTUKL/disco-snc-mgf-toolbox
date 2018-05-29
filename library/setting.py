"""This superclass represents our get_value abstract class"""

from abc import ABC, abstractmethod


class Setting(ABC):
    """Each setting (topology) has to implements methods to obtain
    the bounds"""

    @abstractmethod
    def get_bound(self, theta: float) -> float:
        """
        standard bound

        :param theta: mgf parameter
        """
        pass
