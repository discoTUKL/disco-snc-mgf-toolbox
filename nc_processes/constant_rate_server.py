"""Implemented service classes for different distributions"""

from nc_processes.service import Service


class ConstantRate(Service):
    """Constant rate service"""

    def __init__(self, rate=0.0) -> None:
        self.rate = rate

    def sigma(self, theta=0.0) -> float:
        return 0.0

    def rho(self, theta=1.0) -> float:
        # The minus is important to insure the correct sign
        return -self.rate

    def to_name(self) -> str:
        return self.__class__.__name__

    def to_value(self):
        return "rate=" + str(self.rate)

    def to_name_value(self) -> str:
        """
        :return string
        """
        return self.to_name() + "_" + self.to_value()

    def number_parameters(self) -> int:
        return 1
