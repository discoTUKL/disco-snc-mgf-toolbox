"""Exponentially Bounded Burstiness"""

import math

from utils.exceptions import ParameterOutOfBounds

from nc_arrivals.arrival_distribution import ArrivalDistribution


class EBB(ArrivalDistribution):
    """Exponentially Bounded Burstiness obtained via CCDF"""

    def __init__(self, factor_m: float, decay: float, rho_single: float, discr_time=False, m=1) -> None:
        self.factor_m = factor_m
        self.decay = decay
        self.rho_single = rho_single
        self.discr_time = discr_time
        self.m = m

    def sigma(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta={theta} must be > 0")

        if theta >= self.decay:
            raise ParameterOutOfBounds(f"theta={theta} must " f"be < decay={self.decay}")

        return (self.m / theta) * math.log(1 + self.factor_m * theta / (self.decay - theta))

    def rho(self, theta=0.0) -> float:
        return self.m * self.rho_single

    def is_discrete(self) -> bool:
        return self.discr_time

    def average_rate(self) -> float:
        return self.rho()

    def to_value(self, number=1, show_m=False) -> str:
        if show_m:
            return "M{0}={1}_decay{0}={2}_rho{0}={3}_n{0}={4}".format(str(number), str(self.factor_m), str(self.decay),
                                                                      str(self.rho_single), str(self.m))
        else:
            return "M{0}={1}_decay{0}={2}_rho{0}={3}".format(str(number), str(self.factor_m), str(self.decay),
                                                             str(self.rho_single))
