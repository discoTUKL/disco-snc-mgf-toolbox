""""DNC FIFO Delay bound class."""

from math import ceil

from library.exceptions import ParameterOutOfBounds
from nc_processes.arrival_distribution import TokenBucketConstant
from nc_processes.service import ConstantRate


class DNCFIFODelay(object):
    """DNC FIFO Delay bound class"""

    def __init__(self, token_bucket_constant: TokenBucketConstant,
                 constant_rate: ConstantRate) -> None:
        self.token_bucket_constant = token_bucket_constant
        self.constant_rate = constant_rate

    def bound(self) -> int:
        if self.token_bucket_constant.rho() >= -self.constant_rate.rho():
            raise ParameterOutOfBounds(
                "The arrivals' rho {0} has to be smaller than"
                "the service's rho {1}".format(
                    self.token_bucket_constant.rho(),
                    -self.constant_rate.rho()))

        return int(
            ceil(self.token_bucket_constant.sigma() /
                 (-self.constant_rate.rho())))
