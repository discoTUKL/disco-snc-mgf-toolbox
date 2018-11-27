"""Markov Modulated Processes"""

from math import sqrt

from library.exceptions import ParameterOutOfBounds
from nc_processes.arrival_distribution import ArrivalDistribution


class MMOOCont(ArrivalDistribution):
    """Continuous Markov Modulated On-Off Traffic"""

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
