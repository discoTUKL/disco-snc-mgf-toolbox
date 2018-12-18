""""DNC FIFO Delay class."""

from math import ceil

from utils.exceptions import ParameterOutOfBounds
from nc_processes.regulated_arrivals import TokenBucketConstant
from nc_processes.constant_rate_server import ConstantRate


def fifo_delay(token_bucket_constant: TokenBucketConstant,
               constant_rate: ConstantRate) -> int:
    """DNC FIFO Delay Bound"""
    if token_bucket_constant.rho(1.0) >= constant_rate.rate:
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            " the service's rho {1}".format(token_bucket_constant.rho(1.0),
                                            constant_rate.rate))

    return int(ceil(token_bucket_constant.sigma() / constant_rate.rate))
