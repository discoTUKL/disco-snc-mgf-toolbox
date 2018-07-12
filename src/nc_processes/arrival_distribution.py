"""Implemented arrival classes for different distributions"""

from abc import abstractmethod
from math import exp, log, sqrt

from library.exceptions import ParameterOutOfBounds
from library.helper_functions import is_equal
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
    def to_value(self) -> str:
        pass

    def to_name_value(self) -> str:
        return self.to_name() + self.to_value()

    @abstractmethod
    def number_parameters(self) -> int:
        """
        :return number of parameters that describe the distribution
        """
        pass


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

    def to_value(self) -> str:
        return "mu=" + str(self.mu) + "_lambda=" + str(
            self.lamb) + "_burst=" + str(self.burst) + "_n=" + str(self.n)

    def number_parameters(self) -> int:
        return 3


class EBB(ArrivalDistribution):
    """Exponentially Bounded Burstiness"""

    def __init__(self, prefactor: float, decay: float, rho_single: float,
                 n=1) -> None:
        self.prefactor = prefactor
        self.decay = decay
        self.rho_single = rho_single
        self.n = n

    def sigma(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        if is_equal(theta, self.decay):
            raise ParameterOutOfBounds(
                "theta {0} and decay {1} must be different".format(
                    theta, self.decay))

        theta_decay = theta / self.decay

        return self.n * log(
            (self.prefactor**theta_decay) / (1 - theta_decay)) / theta

    def rho(self, theta=0.0) -> float:
        return self.n * self.rho_single

    def is_discrete(self) -> bool:
        return True

    def to_value(self) -> str:
        return "M=" + str(self.prefactor) + "_b=" + str(
            self.decay) + "_rho=" + str(self.rho_single) + "_n=" + str(self.n)

    def number_parameters(self) -> int:
        return 3


class DM1(ArrivalDistribution):
    """Exponentially distributed packet size."""

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

    def to_value(self) -> str:
        return "lambda=" + str(self.lamb) + "_n=" + str(self.n)

    def number_parameters(self) -> int:
        return 1


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
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        return self.n * self.lamb * (exp(theta * self.packet_size) - 1) / theta

    def is_discrete(self) -> bool:
        return False

    def to_value(self) -> str:
        return "lambda=" + str(self.lamb) + "_b=" + str(
            self.packet_size) + "_n=" + str(self.n)

    def number_parameters(self) -> int:
        return 2
