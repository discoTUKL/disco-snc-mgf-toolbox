"""Implemented service classes for different distributions"""

from abc import abstractmethod
import math

from nc_server.server import Server


class ServerDistribution(Server):
    """Abstract class for arrival processes that are of
    a distinct distribution."""

    @abstractmethod
    def sigma(self, theta: float) -> float:
        """
        sigma(theta)
        :param theta: mgf parameter
        """
        pass

    @abstractmethod
    def rho(self, theta: float) -> float:
        """
        rho(theta)
        :param theta: mgf parameter
        """
        pass

    @abstractmethod
    def average_rate(self) -> float:
        pass

    def to_name(self) -> str:
        return self.__class__.__name__

    def transient_bound(self, theta: float, delta_time: int) -> float:
        if delta_time < 0:
            raise ValueError(f"time is non-negative")

        return math.exp(theta * (-self.rho(theta=theta) * delta_time + self.sigma(theta=theta)))

    @abstractmethod
    def to_value(self, number=1) -> str:
        pass
