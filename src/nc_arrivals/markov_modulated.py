"""Markov Modulated Processes"""

from math import exp, log, sqrt

from nc_arrivals.arrival_distribution import ArrivalDistribution
from utils.exceptions import ParameterOutOfBounds


class MMOOCont(ArrivalDistribution):
    """Continuous-time Markov Modulated On-Off Traffic"""
    def __init__(self, mu: float, lamb: float, peak_rate: float, m=1) -> None:
        self.mu = mu
        self.lamb = lamb
        self.peak_rate = peak_rate
        self.m = m

    def sigma(self, theta=0.0) -> float:
        return 0.0

    def rho(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

        bb = theta * self.peak_rate - self.mu - self.lamb

        return 0.5 * self.m * (bb + sqrt(
            (bb**2) + 4 * self.mu * theta * self.peak_rate)) / theta

    def is_discrete(self) -> bool:
        return False

    def average_rate(self) -> float:
        on_probability = self.mu / (self.lamb + self.mu)
        return self.m * on_probability * self.peak_rate

    def __str__(self) -> str:
        return f"MMOOFluid_mu={self.mu}_lamb={self.lamb}_" \
            f"peak_rate={self.peak_rate}_n={self.m}"

    def to_value(self, number=1, show_m=False) -> str:
        if show_m:
            return "mu{0}={1}_lambda{0}={2}_peak_rate{0}={3}_n{0}={4}".format(
                str(number), str(self.mu), str(self.lamb), str(self.peak_rate),
                str(self.m))
        else:
            return "mu{0}={1}_lambda{0}={2}_peak_rate{0}={3}".format(
                str(number), str(self.mu), str(self.lamb), str(self.peak_rate))


class MMOODisc(ArrivalDistribution):
    """Discrete-time Markov Modulated On-Off Traffic"""
    def __init__(self,
                 stay_on: float,
                 stay_off: float,
                 peak_rate: float,
                 m=1) -> None:
        self.stay_on = stay_on
        self.stay_off = stay_off
        self.peak_rate = peak_rate
        self.m = m

    def sigma(self, theta=0.0) -> float:
        off_on = self.stay_off + self.stay_on * exp(theta * self.peak_rate)
        sqrt_part = sqrt(off_on ** 2 - 4 * (self.stay_off + self.stay_on - 1) *
                         exp(theta * self.peak_rate))

        spectral_density_mmoo = 0.5 * (off_on + sqrt_part)
        v_1 = [spectral_density_mmoo - self.stay_off, 1 - self.stay_off]
        v_2 = [
            (1 - self.stay_on) * exp(theta * self.peak_rate),
            spectral_density_mmoo - self.stay_on * exp(theta * self.peak_rate)
        ]

        if min(v_1) > 0:
            factor = max(v_1) / min(v_1)
        elif min(v_2) > 0:
            factor = max(v_2) / min(v_2)
        else:
            raise ParameterOutOfBounds(f"no factor is > 0")

        return self.m * log(
            exp(theta * self.peak_rate) * factor /
            spectral_density_mmoo) / theta

    def rho(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

        if self.stay_on <= 0.0 or self.stay_on >= 1.0:
            raise ValueError(f"p_stay_on = {self.stay_on} must be in (0,1)")

        if self.stay_off <= 0.0 or self.stay_off >= 1.0:
            raise ValueError(f"p_stay_off = {self.stay_off} must be in (0,1)")

        off_on = self.stay_off + self.stay_on * exp(theta * self.peak_rate)
        sqrt_part = sqrt(off_on ** 2 - 4 * (self.stay_off + self.stay_on - 1) *
                         exp(theta * self.peak_rate))

        rho_mmoo_disc = log(0.5 * (off_on + sqrt_part))

        if rho_mmoo_disc < 0:
            raise ParameterOutOfBounds("rho must be >= 0")

        return rho_mmoo_disc / theta

    def is_discrete(self) -> bool:
        return True

    def average_rate(self) -> float:
        return self.m * (1 - self.stay_off) / (2 - self.stay_off -
                                               self.stay_on) * self.peak_rate

    def __str__(self) -> str:
        return f"MMOODisc_stay_on={self.stay_on}_stay_off={self.stay_off}_" \
            f"peak_rate={self.peak_rate}_n={self.m}"

    def to_value(self, number=1, show_m=False) -> str:
        if show_m:
            return "stay_on{0}={1}_stay_off{0}={2}_peak_rate{0}={3}_" \
                   "n{0}={4}".format(
                str(number), str(self.stay_on), str(self.stay_off),
                str(self.peak_rate), str(self.m))
        else:
            return "stay_on{0}={1}_stay_off{0}={2}_peak_rate{0}={3}".format(
                str(number), str(self.stay_on), str(self.stay_off),
                str(self.peak_rate))
