"""Implemented service classes for different distributions"""

from nc_server.rate_latency_server import RateLatencyServer


class ConstantRateServer(RateLatencyServer):
    """Constant rate service"""

    def __init__(self, rate: float) -> None:
        super().__init__(rate=rate, latency=0.0)

    def sigma(self, theta=0.0) -> float:
        return 0.0

    def rho(self, theta: float) -> float:
        return self.rate

    def average_rate(self) -> float:
        return self.rate

    def __str__(self):
        return f"ConstRate_rate={self.rate}"

    def __repr__(self):
        return str(self)

    def to_value(self, number=1):
        return f"rate{str(number)}={str(self.rate)}"
