"""Implemented arrival classes for different distributions"""

from abc import abstractmethod
from math import exp

from nc_arrivals.arrival import Arrival


class ArrivalDistribution(Arrival):
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
    def is_discrete(self) -> bool:
        """
        :return True if the arrival distribution is discrete, False if not
        """
        pass

    def to_name(self) -> str:
        return self.__class__.__name__

    def transient_bound(self, theta: float, delta_time: int) -> float:
        if delta_time < 0:
            raise ValueError(f"time is non-negative")

        return exp(
            theta *
            (self.rho(theta=theta) * delta_time + self.sigma(theta=theta)))

    @abstractmethod
    def to_value(self, number=1, show_n=False) -> str:
        pass
