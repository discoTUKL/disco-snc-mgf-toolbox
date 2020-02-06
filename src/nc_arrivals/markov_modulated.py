"""Markov Modulated Processes"""

from math import exp, log, sqrt

from nc_arrivals.arrival_distribution import ArrivalDistribution
from utils.exceptions import ParameterOutOfBounds


class MMOOFluid(ArrivalDistribution):
    """Continuous-time Markov Modulated On-Off Traffic"""
    def __init__(self, mu: float, lamb: float, burst: float, n=1) -> None:
        self.mu = mu
        self.lamb = lamb
        self.burst = burst
        self.n = n

    def sigma(self, theta=0.0) -> float:
        return 0.0

    def rho(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

        bb = theta * self.burst - self.mu - self.lamb

        return 0.5 * self.n * (bb + sqrt(
            (bb**2) + 4 * self.mu * theta * self.burst)) / theta

    def is_discrete(self) -> bool:
        return False

    def average_rate(self) -> float:
        on_probability = self.mu / (self.lamb + self.mu)
        return self.n * on_probability * self.burst

    def __str__(self) -> str:
        return f"MMOOFluid_mu={self.mu}_lamb={self.lamb}_" \
            f"burst={self.burst}_n={self.n}"

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "mu{0}={1}_lambda{0}={2}_burst{0}={3}_n{0}={4}".format(
                str(number), str(self.mu), str(self.lamb), str(self.burst),
                str(self.n))
        else:
            return "mu{0}={1}_lambda{0}={2}_burst{0}={3}".format(
                str(number), str(self.mu), str(self.lamb), str(self.burst))


class MMOODisc(ArrivalDistribution):
    """Discrete-time Markov Modulated On-Off Traffic"""
    def __init__(self,
                 stay_on: float,
                 stay_off: float,
                 burst: float,
                 n=1) -> None:
        self.stay_on = stay_on
        self.stay_off = stay_off
        self.burst = burst
        self.n = n

    def sigma(self, theta=0.0) -> float:
        return 0.0

    def rho(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

        if self.stay_on <= 0.0 or self.stay_on >= 1.0:
            raise ValueError(f"p_stay_on = {self.stay_on} must be in (0,1)")

        if self.stay_off <= 0.0 or self.stay_off >= 1.0:
            raise ValueError(f"p_stay_off = {self.stay_off} must be in (0,1)")

        off_on = self.stay_off + self.stay_on * exp(theta * self.burst)
        sqrt_part = sqrt(off_on**2 - 4 * (self.stay_off + self.stay_on - 1) *
                         exp(theta * self.burst))

        rho_mmoo_disc = log(0.5 * (off_on + sqrt_part))

        if rho_mmoo_disc < 0:
            raise ParameterOutOfBounds("rho must be >= 0")

        return rho_mmoo_disc / theta

    def is_discrete(self) -> bool:
        return True

    def average_rate(self) -> float:
        return self.n * (1 - self.stay_off) / (2 - self.stay_off -
                                               self.stay_on) * self.burst

    def __str__(self) -> str:
        return f"MMOODisc_stay_on={self.stay_on}_stay_off={self.stay_off}_" \
            f"burst={self.burst}_n={self.n}"

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "stay_on{0}={1}_stay_off{0}={2}_burst{0}={3}_" \
                   "n{0}={4}".format(
                str(number), str(self.stay_on), str(self.stay_off),
                str(self.burst), str(self.n))
        else:
            return "stay_on{0}={1}_stay_off{0}={2}_burst{0}={3}".format(
                str(number), str(self.stay_on), str(self.stay_off),
                str(self.burst))
