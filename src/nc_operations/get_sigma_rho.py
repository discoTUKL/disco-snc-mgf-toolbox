"""Helper function to compute the rho differences and sum of sigmas quickly"""
from typing import List

from nc_arrivals.arrival import Arrival
from nc_server.server import Server


def get_sigma_rho(arr: Arrival, ser: Server, theta: float, indep=True, p=1.0, q=1.0) -> List[float]:
    if indep:
        return arr.sigma(theta=theta) + ser.sigma(theta=theta), arr.rho(theta=theta) - ser.rho(theta=theta)

    else:
        return arr.sigma(theta=p * theta) + ser.sigma(theta=q * theta), arr.rho(theta=p * theta) - ser.rho(theta=q *
                                                                                                           theta)
