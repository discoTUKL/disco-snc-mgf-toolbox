"""Implements new Lyapunov Deconvolution"""

from math import log

from library.helper_functions import mgf
from library.exceptions import ParameterOutOfBounds
from nc_processes.arrival import Arrival
from nc_processes.service import Service


class DeconvolvePower(Arrival):
    """New Lyapunov Deconvolution Class"""

    def __init__(self, arr: Arrival, ser: Service, l_power=1.0) -> None:
        self.arr = arr
        self.ser = ser
        self.l_power = l_power

        if self.l_power < 1.0:
            self.l_power = 1.0
            # raise ParameterOutOfBounds("l must be >= 1")

    def sigma(self, theta: float) -> float:
        # here, theta can simply be replaced by l * theta
        l_theta = self.l_power * theta

        k_sig = -log(1 - mgf(
            theta=l_theta, x=self.arr.rho(l_theta) + self.ser.rho(l_theta))
                     ) / l_theta

        return self.arr.sigma(l_theta) + self.ser.sigma(l_theta) + k_sig

    def rho(self, theta: float) -> float:
        # here, theta can simply be replaced by l * theta
        l_theta = self.l_power * theta

        if self.arr.rho(l_theta) < 0 or self.ser.rho(l_theta) > 0:
            raise ParameterOutOfBounds("Check RHO_SINGLE's sign")

        if self.arr.rho(l_theta) >= -self.ser.rho(l_theta):
            raise ParameterOutOfBounds(
                "The arrivals' rho has to be smaller than the service's rho")

        return self.arr.rho(l_theta)
