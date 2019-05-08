"""Implemented service classes for different distributions"""

from math import exp

from nc_server.server import Server
from utils.exceptions import ParameterOutOfBounds


class ConstantRateServer(Server):
    """Constant rate service"""

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def sigma(self, theta=0.0) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

        return 0.0

    def rho(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

        return self.rate

    def __str__(self):
        return f"ConstRate_rate={self.rate}"

    def transient_bound(self, theta: float, delta_time: int) -> float:
        if delta_time < 0:
            raise ValueError(f"time is non-negative")

        return exp(-theta * self.rate * delta_time)

    def to_value(self, number=1):
        return "rate{0}={1}".format(str(number), str(self.rate))
