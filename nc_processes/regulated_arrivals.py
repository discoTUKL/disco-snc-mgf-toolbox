"""Abstract Leaky-Bucket class."""

from abc import abstractmethod
from math import erf, exp, inf, log, pi, sqrt

from library.exceptions import ParameterOutOfBounds
from nc_processes.arrival_distribution import ArrivalDistribution


class RegulatedArrivals(ArrivalDistribution):
    """Abstract class for all leaky-bucket classes"""

    def __init__(self, sigma_single=0.0, rho_single=0.0, n=1) -> None:
        self.sigma_single = sigma_single
        self.rho_single = rho_single
        self.n = n

    @abstractmethod
    def sigma(self, theta: float) -> float:
        """
        sigma(theta)
        :param theta: mgf parameter
        """
        pass

    def rho(self, theta=0.0) -> float:
        """
        rho(theta)
        :param theta: mgf parameter
        """
        return self.n * self.rho_single

    def is_discrete(self) -> bool:
        """
        :return True if the arrival distribution is discrete, False if not
        """
        return True

    def to_string(self) -> str:
        """
        :return string
        """

        return "sigma=" + str(self.sigma_single) + "_rho=" + str(
            self.rho) + "_n=" + str(self.n)

    def number_parameters(self) -> int:
        """
        :return number of parameters that describe the distribution
        """
        return 2


class TokenBucketConstant(RegulatedArrivals):
    """Primitive TokenBucket (quasi deterministic and independent of theta)"""

    def __init__(self) -> None:
        super().__init__(self.sigma_single, self.rho_single, self.n)

    def sigma(self, theta=0.0) -> float:
        return self.n * self.sigma_single


class LeakyBucketMassOne(RegulatedArrivals):
    """Leaky Bucket according to Massoulie using directly Lemma 2"""

    def __init__(self, sigma_single=0.0, rho_single=0.0, n=1) -> None:
        super().__init__(sigma_single, rho_single, n)

    def sigma(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        return self.n * log(0.5 * (exp(theta * self.sigma_single) + exp(
            -theta * self.sigma_single))) / theta


class LeakyBucketMassTwo(RegulatedArrivals):
    """Leaky Bucket according to Massoulie after MGF transformation and bound
    on erf() by 1"""

    def __init__(self) -> None:
        super().__init__(self.sigma_single, self.rho_single, self.n)

    def sigma(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds("theta = {0} must be > 0".format(theta))

        try:
            return log(1.0 + sqrt(2 * pi * self.n * (self.sigma_single**2)) *
                       theta * exp(0.5 * self.n * (self.sigma_single**2) *
                                   (theta**2))) / theta
        except OverflowError:
            return inf


class LeakyBucketMassTwoExact(RegulatedArrivals):
    """Exact Leaky Bucket according to Massoulie after MGF transformation"""

    def __init__(self) -> None:
        super().__init__(self.sigma_single, self.rho_single, self.n)

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
