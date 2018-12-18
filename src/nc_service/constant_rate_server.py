"""Implemented service classes for different distributions"""

from nc_service.service import Service
from utils.exceptions import ParameterOutOfBounds


class ConstantRate(Service):
    """Constant rate service"""

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def sigma(self, theta=0.0) -> float:
        # TODO: remove the sign constraints later
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

        return 0.0

    def rho(self, theta: float) -> float:
        # TODO: remove the sign constraints later
        if theta <= 0:
            raise ParameterOutOfBounds(f"theta = {theta} must be > 0")

        return self.rate

    def to_value(self, number=1):
        return "rate{0}={1}".format(str(number), str(self.rate))
