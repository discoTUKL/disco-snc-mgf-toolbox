"""Implemented service classes for different distributions"""

from nc_server.server_distribution import ServerDistribution
from utils.exceptions import ParameterOutOfBounds


class ConstantRateServer(ServerDistribution):
    """Constant rate service"""

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def sigma(self, theta=0.0) -> float:
        return 0.0

    def rho(self, theta: float) -> float:
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

        return self.rate

    def average_rate(self) -> float:
        return self.rate

    def __str__(self):
        return f"ConstRate_rate={self.rate}"

    def to_value(self, number=1):
        return f"rate{str(number)}={str(self.rate)}"
