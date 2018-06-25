"""Implemented arrival classes for different distributions"""

from abc import abstractmethod
from math import log, sqrt

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


class ExponentialArrival(ArrivalDistribution):
    """Exponentially distributed arrivals (parameter lamb)."""

    def __init__(self, lamb=0.0, n=1) -> None:
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
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        if theta >= self.lamb:
            raise ParameterOutOfBounds(
                "theta = {0} must be < lambda = {1}".format(theta, self.lamb))

        return (self.n / theta) * log(self.lamb / (self.lamb - theta))

    def is_discrete(self) -> bool:
        return True

    def to_string(self) -> str:
        return "_lambda=" + str(self.lamb) + "_n=" + str(self.n)

    def number_parameters(self) -> int:
        return 1


class MMOO(ArrivalDistribution):
    """Markov Modulated On Off Traffic"""

    def __init__(self, mu=0.0, lamb=0.0, burst=0.0, n=1) -> None:
        self.mu = mu
        self.lamb = lamb
        self.burst = burst
        self.n = n

    def sigma(self, theta=0.0) -> float:
        return 0.0

    def rho(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        bb = self.mu + self.lamb - theta * self.burst

        return self.n * (-bb + sqrt(
            (bb**2) + 4 * self.mu * theta * self.burst)) / (2 * theta)

    def is_discrete(self) -> bool:
        return False

    def to_string(self) -> str:
        return "mu=" + str(self.mu) + "_lambda=" + str(
            self.lamb) + "_burst=" + str(self.burst) + "_n=" + str(self.n)

    def number_parameters(self) -> int:
        return 3
