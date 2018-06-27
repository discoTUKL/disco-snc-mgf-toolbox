""""DNC FIFO Delay FIFODelay class."""

from math import ceil

from library.exceptions import ParameterOutOfBounds
from nc_processes.regulated_arrivals import TokenBucketConstant
from nc_processes.constant_rate_server import ConstantRate


def FIFODelay(token_bucket_constant: TokenBucketConstant,
              constant_rate: ConstantRate) -> int:
    """DNC FIFO Delay Bound"""
    if token_bucket_constant.rho() >= -constant_rate.rho():
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            " the service's rho {1}".format(token_bucket_constant.rho(),
                                            -constant_rate.rho()))

    return int(ceil(token_bucket_constant.sigma() / (-constant_rate.rho())))
