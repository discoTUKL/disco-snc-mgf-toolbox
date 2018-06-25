"""Implemented service classes for different distributions"""

from abc import abstractmethod

from library.exceptions import ParameterOutOfBounds
from nc_processes.service import Service


class ServiceDistribution(Service):
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
    def to_string(self) -> str:
        """
        :return string
        """
        pass

    def to_string_long(self) -> str:
        """
        :return string
        """
        return self.__class__.__name__ + "_" + self.to_string()

    @abstractmethod
    def number_parameters(self) -> int:
        """
        :return number of parameters that describe the distribution
        """
        pass


class ConstantRate(ServiceDistribution):
    """Constant rate service"""

    def __init__(self, rate=0.0) -> None:
        self.rate = rate

    def sigma(self, theta=0.0) -> float:
        return 0.0

    def rho(self, theta=1.0) -> float:
        if theta <= 0 and theta is not None:
            raise ParameterOutOfBounds(
                "theta = {0} must be > 0".format(theta))

        # The minus is important to insure the correct sign
        return -self.rate

    def to_string(self):
        return "rate=" + str(self.rate)

    def number_parameters(self) -> int:
        return 1
