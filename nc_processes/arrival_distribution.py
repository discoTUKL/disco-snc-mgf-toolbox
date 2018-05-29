"""Implemented arrival classes for different distributions"""

from abc import abstractmethod
from math import erf, exp, inf, log, pi, sqrt

from library.Exceptions import ParameterOutOfBounds
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
    def to_string(self) -> str:
        """
        :return string
        """
        pass

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

    def to_string(self) -> str:
        return self.__class__.__name__ + "_lambda=" + str(
            self.lamb) + "_n=" + str(self.n)

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

    def to_string(self) -> str:
        return self.__class__.__name__ + "_mu=" + str(
            self.mu) + "_lambda=" + str(self.lamb) + "_burst=" + str(
                self.burst) + "_n=" + str(self.n)

    def number_parameters(self) -> int:
        return 3


class LeakyBucketMassOne(ArrivalDistribution):
    """Leaky Bucket according to Massoulie using directly Lemma 2"""

    def __init__(self, sigma_single=0.0, rho_single=0.0, n=1) -> None:
        self.sigma_single = sigma_single
        self.rho_single = rho_single
        self.n = n

    def sigma(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        return self.n * log(0.5 * (exp(theta * self.sigma_single)) +
                            exp(-theta * self.sigma_single)) / theta

    def rho(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        return self.n * self.rho_single

    def to_string(self) -> str:
        return self.__class__.__name__ + "_sigma=" + str(
            self.sigma_single) + "_rho=" + str(self.rho) + "_n=" + str(self.n)

    def number_parameters(self) -> int:
        return 2


class LeakyBucketMassTwo(ArrivalDistribution):
    """Leaky Bucket according to Massoulie after MGF transformation and bound
    on erf() by 1"""

    def __init__(self, sigma_single=0.0, rho_single=0.0, n=1) -> None:
        self.sigma_single = sigma_single
        self.rho_single = rho_single
        self.n = n

    def sigma(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        try:
            return log(1.0 + sqrt(2 * pi * self.n * (self.sigma_single**2)) *
                       theta * exp(0.5 * self.n * (self.sigma_single**2) *
                                   (theta**2))) / theta
        except OverflowError:
            return inf

    def rho(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        return self.n * self.rho_single

    def to_string(self) -> str:
        return self.__class__.__name__ + "_sigma=" + str(
            self.sigma_single) + "_rho=" + str(self.rho) + "_n=" + str(self.n)

    def number_parameters(self) -> int:
        return 2


class LeakyBucketMassTwoExact(ArrivalDistribution):
    """Exact Leaky Bucket according to Massoulie after MGF transformation"""

    def __init__(self, sigma_single=0.0, rho_single=0.0, n=1) -> None:
        self.sigma_single = sigma_single
        self.rho_single = rho_single
        self.n = n

    def sigma(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        try:
            return log(1.0 + sqrt(0.5 * pi * self.n * (self.sigma_single**2)) *
                       theta * exp(0.5 * self.n * (self.sigma_single**2) *
                                   (theta**2)) *
                       erf(1.0 + theta * sqrt(0.5 * self.n *
                                              (self.sigma_single**2)))) / theta
        except OverflowError:
            return inf

    def rho(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        return self.n * self.rho_single

    def to_string(self) -> str:
        return self.__class__.__name__ + "_sigma=" + str(
            self.sigma_single) + "_rho=" + str(self.rho) + "_n=" + str(self.n)

    def number_parameters(self) -> int:
        return 2


class TokenBucketConstant(ArrivalDistribution):
    """Primitive TokenBucket (quasi deterministic and independent of theta)"""

    def __init__(self, sigma_const=0.0, rho_const=0.0, n=1) -> None:
        self.sigma_const = sigma_const
        self.rho_const = rho_const
        self.n = n

    def sigma(self, theta=0.0) -> float:
        return self.n * self.sigma_const

    def rho(self, theta=1.0) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        return self.n * self.rho_const

    def to_string(self) -> str:
        return self.__class__.__name__ + "_sigma=" + str(
            self.sigma_const) + "_rho=" + str(self.rho_const) + "_n=" + str(
                self.n)

    def number_parameters(self) -> int:
        return 2
