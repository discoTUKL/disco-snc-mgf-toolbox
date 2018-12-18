"""Typical Queueing Theory Processes"""

from math import exp, log

from nc_arrivals.arrival_distribution import ArrivalDistribution
from utils.exceptions import ParameterOutOfBounds


class DM1(ArrivalDistribution):
    """Corresponds to D/M/1 queue."""

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
    """Corresponds to M/D/1 queue."""

    def __init__(self, lamb: float, mu: float, n=1) -> None:
        self.lamb = lamb
        self.mu = mu
        self.n = n

    def sigma(self, theta=0.0) -> float:
        return 0.0

    def rho(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

        return (self.n / theta) * self.lamb * (exp(theta / self.mu) - 1)

    def is_discrete(self) -> bool:
        return False

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "lambda{0}={1}_mu{0}={2}_n{0}={3}".format(
                str(number), str(self.lamb), str(self.mu), str(self.n))
        else:
            return "lambda{0}={1}_mu{0}={2}".format(
                str(number), str(self.lamb), str(self.mu))


class MM1(ArrivalDistribution):
    """Corresponds to M/M/1 queue."""

    def __init__(self, lamb: float, mu: float, n=1) -> None:
        self.lamb = lamb
        self.mu = mu
        self.n = n

    def sigma(self, theta=0.0) -> float:
        return 0.0

    def rho(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

        return self.n * self.lamb / (self.mu - theta)

    def is_discrete(self) -> bool:
        return False

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "lambda{0}={1}_mu{0}={2}_n{0}={3}".format(
                str(number), str(self.lamb), str(self.mu), str(self.n))
        else:
            return "lambda{0}={1}_mu{0}={2}".format(
                str(number), str(self.lamb), str(self.mu))
