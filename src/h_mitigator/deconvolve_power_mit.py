"""Implements new Lyapunov Deconvolution"""

from math import exp, log

from nc_arrivals.arrival import Arrival
from nc_server.server import Server
from utils.exceptions import ParameterOutOfBounds


class DeconvolvePowerMit(Arrival):
    """New Lyapunov Deconvolution Class"""

    def __init__(self, arr: Arrival, ser: Server, l_power=1.0) -> None:
        self.arr = arr
        self.ser = ser
        self.l_power = l_power

        if self.l_power < 1.0:
            self.l_power = 1.0
            # raise ParameterOutOfBounds("l must be >= 1")

    def sigma(self, theta: float) -> float:
        # here, theta can simply be replaced by l * theta
        l_theta = self.l_power * theta

        k_sig = -log(
            1 - exp(l_theta *
                    (self.arr.rho(l_theta) - self.ser.rho(l_theta)))) / l_theta

        if self.arr.is_discrete():
            return self.arr.sigma(l_theta) + self.ser.sigma(l_theta) + k_sig
        else:
            return self.arr.sigma(l_theta) + self.ser.sigma(
                l_theta) + self.arr.rho(l_theta) + k_sig

    def rho(self, theta: float) -> float:
        # here, theta can simply be replaced by l * theta
        l_theta = self.l_power * theta

        if self.arr.rho(l_theta) < 0 or self.ser.rho(l_theta) < 0:
            raise ParameterOutOfBounds("Check rho's sign")

        if self.arr.rho(l_theta) >= self.ser.rho(l_theta):
            raise ParameterOutOfBounds(
                "The arrivals' rho has to be smaller than the service's rho")

        return self.arr.rho(l_theta)

    def is_discrete(self):
        return self.arr.is_discrete()
