"""Implemented arrival classes for different distributions"""

from abc import abstractmethod
from math import exp, log

from library.exceptions import ParameterOutOfBounds
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


class DM1(ArrivalDistribution):
    """Exponentially distributed packet size."""

    def __init__(self, lamb: float, n=1) -> None:
        self.lamb = lamb
        self.n = n

    def sigma(self, theta=0.0) -> float:
        """

        :param theta: mgf parameter
        :return:      sigma(theta)
        """
        return 0.0

    def rho(self, theta: float) -> float:
        """
        rho(theta)
        :param theta: mgf parameter
        """
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

        if theta >= self.lamb:
            raise ParameterOutOfBounds(
                f"theta = {theta} must be < lambda = {self.lamb}")

        return (self.n / theta) * log(self.lamb / (self.lamb - theta))

    def is_discrete(self) -> bool:
        return True

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "lambda{0}={1}_n{0}={2}".format(
                str(number), str(self.lamb), str(self.n))
        else:
            return "lambda{0}={1}".format(str(number), str(self.lamb))


class MD1(ArrivalDistribution):
    """Poisson process"""

    def __init__(self, lamb: float, packet_size: float, n=1) -> None:
        self.lamb = lamb
        self.packet_size = packet_size
        self.n = n

    def sigma(self, theta=0.0) -> float:
        return 0.0

    def rho(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

        return (self.n / theta) * self.lamb * (
            exp(theta * self.packet_size) - 1)

    def is_discrete(self) -> bool:
        return False

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "lambda{0}={1}_size{0}={2}_n{0}={3}".format(
                str(number), str(self.lamb), str(self.packet_size),
                str(self.n))
        else:
            return "lambda{0}={1}_size{0}={2}".format(
                str(number), str(self.lamb), str(self.packet_size))
