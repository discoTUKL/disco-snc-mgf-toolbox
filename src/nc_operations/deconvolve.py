"""Output bound class."""

import math

from nc_arrivals.arrival import Arrival
from nc_arrivals.regulated_arrivals import DetermTokenBucket
from nc_server.rate_latency_server import RateLatencyServer
from nc_server.server import Server
from utils.exceptions import ParameterOutOfBounds
from utils.helper_functions import get_q

from nc_operations.stability_check import stability_check


class Deconvolve(Arrival):
    """Deconvolution class."""

    def __init__(self, arr: Arrival, ser: Server, indep=True, p=1.0) -> None:
        self.arr = arr
        self.ser = ser
        self.indep = indep

        if indep:
            self.p = 1.0
            self.q = 1.0
        else:
            self.p = p
            self.q = get_q(p=p)

    def sigma(self, theta: float) -> float:
        """

        :param theta: mgf parameter
        :return:      sigma(theta)
        """
        if isinstance(self.arr, DetermTokenBucket) and isinstance(self.ser, RateLatencyServer):
            return self.arr.burst + self.ser.rate * self.ser.latency

        arr_sigma_p = self.arr.sigma(self.p * theta)
        ser_sigma_q = self.ser.sigma(self.q * theta)

        arr_rho_p = self.arr.rho(self.p * theta)
        k_sig = -math.log(1 - math.exp(theta * (arr_rho_p - self.ser.rho(self.q * theta)))) / theta

        if self.arr.is_discrete():
            return arr_sigma_p + ser_sigma_q + k_sig
        else:
            return arr_sigma_p + ser_sigma_q + arr_rho_p + k_sig

    def rho(self, theta: float) -> float:
        """

        :param theta: mgf parameter
        :return: rho(theta)
        """
        if isinstance(self.arr, DetermTokenBucket) and isinstance(self.ser, RateLatencyServer):
            stability_check(arr=self.arr, ser=self.ser, theta=theta)
            return self.arr.arr_rate

        arr_rho_p = self.arr.rho(self.p * theta)

        if arr_rho_p < 0 or self.ser.rho(self.q * theta) < 0:
            raise ParameterOutOfBounds("The rhos must be >= 0")

        stability_check(arr=self.arr, ser=self.ser, theta=theta, indep=self.indep, p=self.p, q=self.q)

        return arr_rho_p

    def is_discrete(self):
        return self.arr.is_discrete()
