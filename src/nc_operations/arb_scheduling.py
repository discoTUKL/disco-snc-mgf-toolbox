"""Implements all network operations in the sigma-rho calculus."""

from nc_arrivals.arrival import Arrival
from nc_arrivals.regulated_arrivals import DetermTokenBucket
from nc_server.rate_latency_server import RateLatencyServer
from nc_server.server import Server
from utils.exceptions import ParameterOutOfBounds
from utils.helper_functions import get_q


class LeftoverARB(Server):
    """Class to compute the leftover service for ARB."""
    def __init__(self,
                 ser: Server,
                 cross_arr: Arrival,
                 indep=True,
                 p=1.0) -> None:
        self.ser = ser
        self.cross_arr = cross_arr
        self.indep = indep

        if indep:
            self.p = 1.0
            self.q = 1.0
        else:
            self.p = p
            self.q = get_q(p=p)

    def sigma(self, theta):
        if isinstance(self.ser, RateLatencyServer) and isinstance(
                self.cross_arr, DetermTokenBucket):
            return (self.cross_arr.burst + self.ser.rate * self.ser.latency
                    ) / (self.ser.rate - self.cross_arr.arr_rate)

        return self.ser.sigma(theta=self.q * theta) + self.cross_arr.sigma(
            theta=self.p * theta)

    def rho(self, theta):
        if isinstance(self.ser, RateLatencyServer) and isinstance(
                self.cross_arr, DetermTokenBucket):
            return self.ser.rate - self.cross_arr.arr_rate

        arr_rho_p_theta = self.cross_arr.rho(theta=self.p * theta)
        ser_rho_q_theta = self.ser.rho(theta=self.q * theta)

        if ser_rho_q_theta < 0 or arr_rho_p_theta < 0:
            raise ParameterOutOfBounds("The rhos must be >= 0")

        return ser_rho_q_theta - arr_rho_p_theta
