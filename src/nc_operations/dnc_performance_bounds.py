""""DNC FIFO Delay class."""

from math import ceil

from nc_arrivals.regulated_arrivals import TokenBucketConstant
from nc_server.constant_rate_server import ConstantRateServer
from utils.exceptions import ParameterOutOfBounds


def fifo_delay(token_bucket_constant: TokenBucketConstant,
               constant_rate: ConstantRateServer) -> int:
    """DNC FIFO Delay Bound"""
    if token_bucket_constant.rho(1.0) >= constant_rate.rate:
        raise ParameterOutOfBounds(
            "The arrivals' rho {0} has to be smaller than"
            " the service's rho {1}".format(
                token_bucket_constant.rho(1.0), constant_rate.rate))

    return int(ceil(token_bucket_constant.sigma() / constant_rate.rate))
