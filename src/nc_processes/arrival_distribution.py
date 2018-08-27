"""Implemented arrival classes for different distributions"""

from abc import abstractmethod
from math import exp, log, sqrt

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


class MMOO(ArrivalDistribution):
    """Markov Modulated On-Off Traffic"""

    def __init__(self, mu: float, lamb: float, burst: float, n=1) -> None:
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

        return (self.n / (2 * theta)) * (-bb + sqrt(
            (bb**2) + 4 * self.mu * theta * self.burst))

    def is_discrete(self) -> bool:
        return False

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "mu{0}={1}_lambda{0}={2}_burst{0}={3}_n{0}={4}".format(
                str(number), str(self.mu), str(self.lamb), str(self.burst),
                str(self.n))
        else:
            return "mu{0}={1}_lambda{0}={2}_burst{0}={3}".format(
                str(number), str(self.mu), str(self.lamb), str(self.burst))


class EBB(ArrivalDistribution):
    """Exponentially Bounded Burstiness"""

    def __init__(self, pre_m: float, decay: float, rho_single: float,
                 n=1) -> None:
        self.pre_m = pre_m
        self.decay = decay
        self.rho_single = rho_single
        self.n = n

    def sigma(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        if theta >= self.decay:
            raise ParameterOutOfBounds("theta {0} must be < decay {1}".format(
                theta, self.decay))

        theta_over_decay = theta / self.decay

        return (self.n / theta) * log(
            (self.pre_m**theta_over_decay) / (1 - theta_over_decay))

    def rho(self, theta=0.0) -> float:
        return self.n * self.rho_single

    def is_discrete(self) -> bool:
        return False

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "M{0}={1}_decay{0}={2}_rho{0}={3}_n{0}={4}".format(
                str(number), str(self.pre_m), str(self.decay),
                str(self.rho_single), str(self.n))
        else:
            return "M{0}={1}_decay{0}={2}_rho{0}={3}".format(
                str(number), str(self.pre_m), str(self.decay),
                str(self.rho_single))


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
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        if theta >= self.lamb:
            raise ParameterOutOfBounds(
                "theta = {0} must be < lambda = {1}".format(theta, self.lamb))

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
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

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
