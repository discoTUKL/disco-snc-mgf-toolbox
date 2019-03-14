"""Helper function to compute the rho differences and sum of sigmas quickly"""

from nc_arrivals.arrival import Arrival
from nc_service.service import Service
from utils.helper_functions import get_q


def get_sigma_rho(arr: Arrival, ser: Service, theta: float, indep: bool,
                  p: float) -> (float, float):
    if indep:
        return arr.sigma(theta=theta) + ser.sigma(theta=theta), arr.rho(
            theta=theta) - ser.rho(theta=theta)

    else:
        q = get_q(p=p, indep=indep)

        return arr.sigma(theta=p * theta) + ser.sigma(
            theta=q * theta), arr.rho(theta=p * theta) - ser.rho(
                theta=q * theta)
