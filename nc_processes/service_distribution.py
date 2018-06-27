"""Implemented service classes for different distributions"""

from library.exceptions import ParameterOutOfBounds
from nc_processes.service import Service


class ConstantRate(Service):
    """Constant rate service"""

    def __init__(self, rate=0.0) -> None:
        self.rate = rate

    def sigma(self, theta=0.0) -> float:
        return 0.0

    def rho(self, theta=1.0) -> float:
        if theta <= 0 and theta is not None:
            raise ParameterOutOfBounds(
                "theta = {0} must be > 0".format(theta))

        # The minus is important to insure the correct sign
        return -self.rate

    def to_string(self):
        return "rate=" + str(self.rate)

    def number_parameters(self) -> int:
        return 1
