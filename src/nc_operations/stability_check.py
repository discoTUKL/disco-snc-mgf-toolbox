"""Check the system's stability"""

from nc_arrivals.arrival import Arrival
from nc_server.server import Server
from utils.exceptions import ParameterOutOfBounds


def stability_check(arr: Arrival,
                    ser: Server,
                    theta: float,
                    indep=True,
                    p=1.0,
                    q=1.0) -> None:
    if indep:
        # if (arr.rho(theta=theta) < 0 or ser.rho(theta=theta) < 0
        #         or arr.sigma(theta=theta) < 0 or ser.sigma(theta=theta) < 0):
        #     raise ParameterOutOfBounds("parameters must be positive")

        if arr.rho(theta=theta) >= ser.rho(theta=theta):
            raise ParameterOutOfBounds(
                f"The arrivals' rho={arr.rho(theta=theta)} has to be "
                f"smaller than the service's rho={ser.rho(theta=theta)}")
    else:
        # if (arr.rho(theta=p * theta) < 0 or ser.rho(theta=q * theta) < 0
        #         or arr.sigma(theta=p * theta) < 0
        #         or ser.sigma(theta=q * theta) < 0):
        #     raise ParameterOutOfBounds("parameters must be positive")

        if arr.rho(theta=p * theta) >= ser.rho(theta=q * theta):
            raise ParameterOutOfBounds(
                f"The arrivals' rho={arr.rho(theta=p * theta)} has to be "
                f"smaller than the service's rho={ser.rho(theta=q * theta)}")
