"""Implemented arrival classes for different distributions"""

from abc import abstractmethod

from nc_processes.arrival import Arrival


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

    @abstractmethod
    def to_value(self, number=1, show_n=False) -> str:
        pass
