"""Implements all network operations in the sigma-rho calculus."""

from typing import List

from nc_arrivals.arrival import Arrival
from nc_server.server import Server
from utils.exceptions import ParameterOutOfBounds, WrongDimension
from utils.helper_functions import get_q


class LeftoverARB(Server):
    """Class to compute the leftover service."""
    def __init__(self, ser: Server, arr: Arrival, indep=True, p=1.0) -> None:
        self.ser = ser
        self.arr = arr
        self.indep = indep

        if indep:
            self.p = 1.0
        else:
            self.p = p

        self.q = get_q(p=p, indep=indep)

    def sigma(self, theta):
        return self.ser.sigma(self.q * theta) + self.arr.sigma(self.p * theta)

    def rho(self, theta):
        arr_rho_p_theta = self.arr.rho(self.p * theta)
        ser_rho_q_theta = self.ser.rho(self.q * theta)

        if ser_rho_q_theta < 0 or arr_rho_p_theta < 0:
            raise ParameterOutOfBounds("The rhos must be >= 0")

        return ser_rho_q_theta - arr_rho_p_theta
