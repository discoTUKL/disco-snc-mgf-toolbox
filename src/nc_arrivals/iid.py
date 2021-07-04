"""Typical Queueing Theory Processes"""

from math import erf, exp, log, pi, sqrt

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

    def average_rate(self) -> float:
        return self.n / self.lamb

    def __str__(self) -> str:
        return f"D/M/1_lambda={self.lamb}_n={self.n}"

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "lambda{0}={1}_n{0}={2}".format(str(number), str(self.lamb),
                                                   str(self.n))
        else:
            return "lambda{0}={1}".format(str(number), str(self.lamb))


class DGamma1(ArrivalDistribution):
    """Corresponds to D/Gamma/1 queue."""
    def __init__(self, alpha_shape: float, beta_rate: float, n=1) -> None:
        self.alpha_shape = alpha_shape
        self.beta_rate = beta_rate
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

        if theta >= self.beta_rate:
            raise ParameterOutOfBounds(
                f"theta = {theta} must be < beta = {self.beta_rate}")

        return (self.n * self.alpha_shape / theta) * log(
            self.beta_rate / (self.beta_rate - theta))

    def is_discrete(self) -> bool:
        return True

    def average_rate(self) -> float:
        return self.n * self.alpha_shape / self.beta_rate

    def __str__(self) -> str:
        return f"D/Gamma/1_alpha={self.alpha_shape}_" \
               f"beta={self.beta_rate}_n={self.n}"

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "alpha{0}={1}_beta{0}={2}_n{0}={3}".format(
                str(number), str(self.alpha_shape), str(self.beta_rate),
                str(self.n))
        else:
            return "alpha{0}={1}_beta{0}={2}".format(str(number),
                                                     str(self.alpha_shape),
                                                     str(self.beta_rate))


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

    def average_rate(self):
        return self.n * self.lamb / self.mu

    def __str__(self) -> str:
        return f"M/D/1_lambda={self.lamb}_mu={self.mu}_n={self.n}"

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "lambda{0}={1}_mu{0}={2}_n{0}={3}".format(
                str(number), str(self.lamb), str(self.mu), str(self.n))
        else:
            return "lambda{0}={1}_mu{0}={2}".format(str(number),
                                                    str(self.lamb),
                                                    str(self.mu))


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

        if theta >= self.mu:
            raise ParameterOutOfBounds(f"theta = {theta} must"
                                       f"be < mu = {self.mu}")

        return self.n * self.lamb / (self.mu - theta)

    def is_discrete(self) -> bool:
        return False

    def average_rate(self):
        return self.n * self.lamb / self.mu

    def __str__(self) -> str:
        return f"M/M/1_lambda={self.lamb}_mu={self.mu}_n={self.n}"

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "lambda{0}={1}_mu{0}={2}_n{0}={3}".format(
                str(number), str(self.lamb), str(self.mu), str(self.n))
        else:
            return "lambda{0}={1}_mu{0}={2}".format(str(number),
                                                    str(self.lamb),
                                                    str(self.mu))


class DPoisson1(ArrivalDistribution):
    """Corresponds to D/Poisson/1 queue."""
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

        return (self.n / theta) * self.lamb * (exp(theta) - 1)

    def is_discrete(self) -> bool:
        return True

    def average_rate(self) -> float:
        return self.n * self.lamb

    def __str__(self) -> str:
        return f"Poisson_lambda={self.lamb}_n={self.n}"

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "lambda{0}={1}_n{0}={2}".format(str(number), str(self.lamb),
                                                   str(self.n))
        else:
            return "lambda{0}={1}".format(str(number), str(self.lamb))


class DWeibull1(ArrivalDistribution):
    """Corresponds to D/Weibull/1 queue."""
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

        sigma = self.lamb / sqrt(2)

        error_part = erf(sigma * theta / sqrt(2)) + 1

        return self.n * log(1 + sigma * theta * exp(0.5 * (sigma * theta)**2) *
                            sqrt(0.5 * pi) * error_part) / theta

    def is_discrete(self) -> bool:
        return True

    def average_rate(self) -> float:
        sigma = self.lamb / sqrt(2)

        return self.n * sigma * sqrt(0.5 * pi)

    def __str__(self) -> str:
        return f"Weibull_lambda={self.lamb}_n={self.n}"

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "lambda{0}={1}_n{0}={2}".format(str(number), str(self.lamb),
                                                   str(self.n))
        else:
            return "lambda{0}={1}".format(str(number), str(self.lamb))
