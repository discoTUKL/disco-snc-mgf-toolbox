"""Check the system's stability"""

from nc_arrivals.arrival import Arrival
from nc_service.service import Service
from utils.exceptions import ParameterOutOfBounds
from utils.helper_functions import get_q


def stability_check(arr: Arrival, ser: Service, theta: float, indep: bool,
                    p: float) -> None:
    if indep:
        if arr.rho(theta=theta) >= ser.rho(theta=theta):
            raise ParameterOutOfBounds(
                f"The arrivals' rho={arr.rho(theta=theta)} has to be "
                f"smaller than the service's rho={ser.rho(theta=theta)}")
    else:
        q = get_q(p=p, indep=indep)

        if arr.rho(theta=p * theta) >= ser.rho(theta=q * theta):
            raise ParameterOutOfBounds(
                f"The arrivals' rho={arr.rho(theta=p * theta)} has to be "
                f"smaller than the service's rho={ser.rho(theta=q * theta)}")
