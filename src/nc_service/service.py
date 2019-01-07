"""Implemented service class"""

from abc import abstractmethod, ABC


class Service(ABC):
    """Abstract Service class"""

    @abstractmethod
    def sigma(self, theta: float) -> float:
        """Sigma method"""
        pass

    @abstractmethod
    def rho(self, theta: float) -> float:
        """Rho method"""
        pass
