"""Convolution class."""

from math import exp, log
from warnings import warn

from nc_server.rate_latency_server import RateLatencyServer
from nc_server.server import Server
from utils.exceptions import ParameterOutOfBounds
from utils.helper_functions import get_q, is_equal


class Convolve(Server):
    """Convolution class."""
    def __init__(self, ser1: Server, ser2: Server, indep=True, p=1.0) -> None:
        self.ser1 = ser1
        self.ser2 = ser2
        self.indep = indep

        if indep:
            self.p = 1.0
            self.q = 1.0
        else:
            self.p = p
            self.q = get_q(p=p)

    def sigma(self, theta: float) -> float:
        if isinstance(self.ser1, RateLatencyServer) and isinstance(
                self.ser2, RateLatencyServer):
            return self.ser1.rate * self.ser1.latency + self.ser2.rate * self.ser2.latency

        ser_1_sigma_p = self.ser1.sigma(self.p * theta)
        ser_2_sigma_q = self.ser2.sigma(self.q * theta)

        ser_1_rho_p = self.ser1.rho(self.p * theta)
        ser_2_rho_q = self.ser2.rho(self.q * theta)

        if not is_equal(ser_1_rho_p, ser_2_rho_q):
            k_sig = -log(1 -
                         exp(-theta * abs(ser_1_rho_p - ser_2_rho_q))) / theta

            return ser_1_sigma_p + ser_2_sigma_q + k_sig

        else:
            return ser_1_sigma_p + ser_2_sigma_q

    def rho(self, theta: float) -> float:
        if isinstance(self.ser1, RateLatencyServer) and isinstance(
                self.ser2, RateLatencyServer):
            return min(self.ser1.rate, self.ser2.rate)

        ser_1_rho_p = self.ser1.rho(self.p * theta)
        ser_2_rho_q = self.ser2.rho(self.q * theta)

        if ser_1_rho_p < 0 or ser_2_rho_q < 0:
            raise ParameterOutOfBounds("The rhos must be > 0")

        if not is_equal(ser_1_rho_p, ser_2_rho_q):
            return min(ser_1_rho_p, ser_2_rho_q)

        else:
            warn("better use ConvolveRateReduct() for equal rhos")
            return ser_1_rho_p - 1 / theta


class ConvolveRateReduction(Server):
    """Convolution class."""
    def __init__(self,
                 ser1: Server,
                 ser2: Server,
                 delta: float,
                 indep=True,
                 p=1.0) -> None:
        self.ser1 = ser1
        self.ser2 = ser2
        self.indep = indep

        if indep:
            self.p = 1.0
            self.q = 1.0
        else:
            self.p = p
            self.q = get_q(p=p)

        self.delta = delta

    def sigma(self, theta: float) -> float:
        if isinstance(self.ser1, RateLatencyServer) and isinstance(
                self.ser2, RateLatencyServer):
            return self.ser1.rate * self.ser1.latency + self.ser2.rate * self.ser2.latency

        ser_1_sigma_p = self.ser1.sigma(self.p * theta)
        ser_2_sigma_q = self.ser2.sigma(self.q * theta)

        ser_1_rho_p = self.ser1.rho(self.p * theta)
        ser_2_rho_q = self.ser2.rho(self.q * theta)

        if not is_equal(ser_1_rho_p, ser_2_rho_q):
            k_sig = -log(1 -
                         exp(-theta * abs(ser_1_rho_p - ser_2_rho_q))) / theta

            return ser_1_sigma_p + ser_2_sigma_q + k_sig

        else:
            return ser_1_sigma_p + ser_2_sigma_q - log(1 - exp(-theta *
                                                               self.delta))

    def rho(self, theta: float) -> float:
        if isinstance(self.ser1, RateLatencyServer) and isinstance(
                self.ser2, RateLatencyServer):
            return min(self.ser1.rate, self.ser2.rate)

        ser_1_rho_p = self.ser1.rho(self.p * theta)
        ser_2_rho_q = self.ser2.rho(self.q * theta)

        if ser_1_rho_p < 0 or ser_2_rho_q < 0:
            raise ParameterOutOfBounds("The rhos must be > 0")

        if not is_equal(ser_1_rho_p, ser_2_rho_q):
            return min(ser_1_rho_p, ser_2_rho_q)

        else:
            if self.delta < 0 or ser_1_rho_p - self.delta <= 0:
                raise ParameterOutOfBounds("Residual rate must be > 0")

            return ser_1_rho_p - self.delta
