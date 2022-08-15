""""DNC FIFO Delay class."""

from math import inf

from nc_arrivals.regulated_arrivals import DetermTokenBucket
from nc_operations.stability_check import stability_check
from nc_server.rate_latency_server import RateLatencyServer
from utils.exceptions import ParameterOutOfBounds


def dnc_delay(tb: DetermTokenBucket, rl: RateLatencyServer) -> float:
    """DNC FIFO Delay Bound"""
    try:
        stability_check(arr=tb, ser=rl, theta=1.0)
    except ParameterOutOfBounds:
        return inf

    return rl.latency + tb.burst / rl.rate
