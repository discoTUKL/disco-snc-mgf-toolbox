"""Implemented service classes for different distributions"""

from nc_processes.service import Service


class ConstantRate(Service):
    """Constant rate service"""

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def sigma(self, theta: float = 0.0) -> float:
        return 0.0

    def rho(self, theta: float) -> float:
        # The minus is important to insure the correct sign
        return -self.rate

    def to_value(self):
        return "rate={0}".format(str(self.rate))
