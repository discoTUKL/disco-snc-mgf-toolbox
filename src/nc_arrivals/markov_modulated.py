"""Markov Modulated Processes"""

import math

from utils.exceptions import IllegalArgumentError, ParameterOutOfBounds

from nc_arrivals.arrival_distribution import ArrivalDistribution


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

        return 0.5 * self.m * (bb + math.sqrt((bb**2) + 4 * self.mu * theta * self.peak_rate)) / theta

    def is_discrete(self) -> bool:
        return False

    def average_rate(self) -> float:
        on_probability = self.mu / (self.lamb + self.mu)
        return self.m * on_probability * self.peak_rate

    def __str__(self) -> str:
        return f"MMOOFluid_mu={self.mu}_lamb={self.lamb}_peak_rate={self.peak_rate}_n={self.m}"

    def to_value(self, number=1, show_m=False) -> str:
        if show_m:
            return "mu{0}={1}_lambda{0}={2}_peak_rate{0}={3}_n{0}={4}".format(str(number), str(self.mu), str(self.lamb),
                                                                              str(self.peak_rate), str(self.m))
        else:
            return "mu{0}={1}_lambda{0}={2}_peak_rate{0}={3}".format(str(number), str(self.mu), str(self.lamb),
                                                                     str(self.peak_rate))


class MMOODisc(ArrivalDistribution):
    """Discrete-time Markov Modulated On-Off Traffic"""

    def __init__(self, stay_on: float, stay_off: float, peak_rate: float, m=1) -> None:
        self.stay_on = stay_on
        self.stay_off = stay_off
        self.peak_rate = peak_rate
        self.m = m

    def sigma(self, theta=0.0) -> float:
        off_on = self.stay_off + self.stay_on * math.exp(theta * self.peak_rate)
        sqrt_part = math.sqrt(off_on**2 - 4 * (self.stay_off + self.stay_on - 1) * math.exp(theta * self.peak_rate))

        spectral_rad = 0.5 * (off_on + sqrt_part)

        eigen_vec = [1 - self.stay_off, spectral_rad - self.stay_off]

        if eigen_vec[0] * eigen_vec[1] > 0:
            if eigen_vec[0] > 0:
                factor = max(eigen_vec) / min(eigen_vec)
            else:
                factor = min(eigen_vec) / max(eigen_vec)
        else:
            raise ParameterOutOfBounds(f"no factor is > 0")

        return self.m * math.log(math.exp(theta * self.peak_rate) * factor / spectral_rad) / theta

    def rho(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

        if self.stay_on <= 0.0 or self.stay_on >= 1.0:
            raise IllegalArgumentError(f"p_stay_on = {self.stay_on} must " f"be in (0,1)")

        if self.stay_off <= 0.0 or self.stay_off >= 1.0:
            raise IllegalArgumentError(f"p_stay_off = {self.stay_off} must " f"be in (0,1)")

        off_on = self.stay_off + self.stay_on * math.exp(theta * self.peak_rate)
        sqrt_part = math.sqrt(off_on**2 - 4 * (self.stay_off + self.stay_on - 1) * math.exp(theta * self.peak_rate))

        spectral_rad = 0.5 * (off_on + sqrt_part)

        rho_mmoo_disc = self.m * math.log(spectral_rad) / theta

        if rho_mmoo_disc < 0:
            raise ParameterOutOfBounds("rho must be >= 0")

        return rho_mmoo_disc

    def is_discrete(self) -> bool:
        return True

    def average_rate(self) -> float:
        p_on = (1 - self.stay_off) / (2 - self.stay_off - self.stay_on)

        return self.m * p_on * self.peak_rate

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
            return "stay_on{0}={1}_stay_off{0}={2}_peak_rate{0}={3}".format(str(number), str(self.stay_on),
                                                                            str(self.stay_off), str(self.peak_rate))
