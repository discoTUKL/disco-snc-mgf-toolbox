"""Implements new Lyapunov NC_Operations.Output Bound"""

from math import exp, inf

from library.exceptions import ParameterOutOfBounds
from library.helper_functions import is_equal
from nc_processes.arrival import Arrival
from nc_processes.service import Service


class OutputLya(object):
    """New Lyapunov Output Class"""

    def __init__(self, arr: Arrival, ser: Service, l_lya=1.0) -> None:
        self.arr = arr
        self.ser = ser
        self.l_lya = l_lya

        if self.l_lya < 1.0:
            self.l_lya = 1.0
            # raise ParameterOutOfBounds("l must be >= 1")

    def bound(self, theta: float, delta_time: int) -> float:
        """Implements stationary bound method"""

        l_theta = self.l_lya * theta

        if self.arr.rho(l_theta) >= -self.ser.rho(l_theta):
            raise ParameterOutOfBounds(
                "The arrivals' RHO_SINGLE {0} has to be smaller than"
                "the service's RHO_SINGLE {1}".format(
                    self.arr.rho(theta), -self.ser.rho(theta)))

        sigma_l_arr_ser = self.arr.sigma(l_theta) + self.ser.sigma(l_theta)
        rho_l_arr_ser = self.arr.rho(l_theta) + self.ser.rho(l_theta)

        numerator = exp(
            theta * (self.arr.rho(l_theta) * delta_time + sigma_l_arr_ser))
        denominator = (1 - exp(l_theta * rho_l_arr_ser))**(1 / self.l_lya)

        if is_equal(denominator, 0):
            return inf

        return numerator / denominator

    def bound_t(self, theta: float, tt: int, ss: int) -> float:
        """Implements time dependent method"""

        l_theta = self.l_lya * theta

        sigma_l_arr_ser = self.arr.sigma(l_theta) + self.ser.sigma(l_theta)
        rho_l_arr_ser = self.arr.rho(l_theta) + self.ser.rho(l_theta)

        if is_equal(self.arr.rho(l_theta), -self.ser.rho(l_theta)):
            return exp(theta * (self.arr.rho(l_theta) *
                                (tt - ss) + sigma_l_arr_ser)) * (ss + 1)**(
                                    1 / self.l_lya)

        elif self.arr.rho(l_theta) > -self.ser.rho(l_theta):
            numerator = exp(
                theta * (self.arr.rho(l_theta) * tt +
                         self.ser.rho(l_theta) * ss + sigma_l_arr_ser))
            denominator = 1 - exp(-l_theta * rho_l_arr_ser)**(1 / self.l_lya)

            return numerator / denominator

        else:
            return self.bound(theta=theta, delta_time=tt - ss)


class DelayProbLya(object):
    """New Lyapunov nc_operations.DelayProb Class"""

    def __init__(self, arr: Arrival, ser: Service, l_lya=1.0) -> None:
        self.arr = arr
        self.ser = ser
        self.l_lya = l_lya

        if self.l_lya < 1.0:
            self.l_lya = 1.0
            # raise ParameterOutOfBounds("l must be >= 1")

    def bound(self, theta: float, delay: int) -> float:
        """Implements stationary bound method"""

        l_theta = self.l_lya * theta

        if self.arr.rho(l_theta) >= -self.ser.rho(l_theta):
            raise ParameterOutOfBounds(
                "The arrivals' RHO_SINGLE {0} has to be smaller than"
                "the service's RHO_SINGLE {1}".format(
                    self.arr.rho(theta), -self.ser.rho(theta)))

        sigma_l_arr_ser = self.arr.sigma(l_theta) + self.ser.sigma(l_theta)
        rho_l_arr_ser = self.arr.rho(l_theta) + self.ser.rho(l_theta)

        numerator = exp(
            theta * (self.ser.rho(l_theta) * delay + sigma_l_arr_ser))
        denominator = (1 - exp(l_theta * rho_l_arr_ser))**(1 / self.l_lya)

        return numerator / denominator

    def bound_t(self, theta: float, delay: int, tt: int) -> float:
        """Implements time dependent method"""

        l_theta = self.l_lya * theta

        sigma_l_arr_ser = self.arr.sigma(l_theta) + self.ser.sigma(l_theta)
        rho_l_arr_ser = self.arr.rho(l_theta) + self.ser.rho(l_theta)

        if is_equal(self.arr.rho(l_theta), -self.ser.rho(l_theta)):
            return exp(theta *
                       (self.ser.rho(l_theta) * delay + sigma_l_arr_ser)) * (
                           tt + 1)**(1 / self.l_lya)

        elif self.arr.rho(l_theta) > -self.ser.rho(l_theta):
            numerator = exp(
                theta * (self.arr.rho(l_theta) * tt + self.ser.rho(l_theta) *
                         (tt + delay) + sigma_l_arr_ser))
            denominator = 1 - exp(-l_theta * rho_l_arr_ser)**(1 / self.l_lya)

            return numerator / denominator

        else:
            return self.bound(theta=theta, delay=delay)


class OutputLyaDiscretized(object):
    """New Lyapunov Output Class for continuous processes"""

    def __init__(self, arr: Arrival, ser: Service, l_lya=1.0) -> None:
        self.arr = arr
        self.ser = ser
        self.l_lya = l_lya

        if self.l_lya < 1.0:
            self.l_lya = 1.0
            # raise ParameterOutOfBounds("l must be >= 1")

    def bound(self, theta: float, delta_time: int) -> float:
        """Implements stationary bound method"""

        l_theta = self.l_lya * theta

        if self.arr.rho(l_theta) >= -self.ser.rho(l_theta):
            raise ParameterOutOfBounds(
                "The arrivals' RHO_SINGLE {0} has to be smaller than"
                "the service's RHO_SINGLE {1}".format(
                    self.arr.rho(theta), -self.ser.rho(theta)))

        sigma_l_arr_ser = self.arr.sigma(l_theta) + self.ser.sigma(l_theta)
        rho_l_arr_ser = self.arr.rho(l_theta) + self.ser.rho(l_theta)

        numerator = exp(theta * (self.arr.rho(l_theta) *
                                 (delta_time + 1) + sigma_l_arr_ser))
        denominator = (1 - exp(l_theta * rho_l_arr_ser))**(1 / self.l_lya)

        if is_equal(denominator, 0):
            return inf

        return numerator / denominator
