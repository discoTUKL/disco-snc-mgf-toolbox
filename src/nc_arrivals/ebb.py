"""Exponentially Bounded Burstiness"""

from math import log

from nc_arrivals.arrival_distribution import ArrivalDistribution
from utils.exceptions import ParameterOutOfBounds


class EBB(ArrivalDistribution):
    """Exponentially Bounded Burstiness obtained via CCDF"""
    def __init__(self,
                 factor_m: float,
                 decay: float,
                 rho_single: float,
                 discr_time=False,
                 n=1) -> None:
        self.factor_m = factor_m
        self.decay = decay
        self.rho_single = rho_single
        self.discr_time = discr_time
        self.n = n

    def sigma(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta={theta} must be > 0")

        if theta >= self.decay:
            raise ParameterOutOfBounds(f"theta={theta} must "
                                       f"be < decay={self.decay}")

        theta_over_decay = theta / self.decay

        return (self.n / theta) * log(1 + self.factor_m * theta /
                                      (self.decay - theta))

    def rho(self, theta=0.0) -> float:
        return self.n * self.rho_single

    def is_discrete(self) -> bool:
        return self.discr_time

    def average_rate(self) -> float:
        return self.rho()

    def to_value(self, number=1, show_n=False) -> str:
        if show_n:
            return "M{0}={1}_decay{0}={2}_rho{0}={3}_n{0}={4}".format(
                str(number), str(self.factor_m), str(self.decay),
                str(self.rho_single), str(self.n))
        else:
            return "M{0}={1}_decay{0}={2}_rho{0}={3}".format(
                str(number), str(self.factor_m), str(self.decay),
                str(self.rho_single))
