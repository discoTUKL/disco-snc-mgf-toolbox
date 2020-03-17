"""Implemented service classes for different distributions"""

from nc_server.server_distribution import ServerDistribution


class RateLatencyServer(ServerDistribution):
    """Constant rate service"""

    def __init__(self, rate: float, latency: float) -> None:
        self.rate = rate
        self.latency = latency

    def sigma(self, theta: float) -> float:
        return self.latency

    def rho(self, theta: float) -> float:
        return self.rate

    def average_rate(self) -> float:
        return self.rate

    def __str__(self):
        return f"RateLatency_rate={self.rate}_latency={self.latency}"

    def to_value(self, number=1):
        return f"rate{str(number)}={str(self.rate)}"
