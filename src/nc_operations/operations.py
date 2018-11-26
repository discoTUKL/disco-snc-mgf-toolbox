"""Implements all network operations in the sigma-rho calculus."""

from math import exp, log
from typing import List

from library.exceptions import ParameterOutOfBounds
from library.helper_functions import get_p_n, get_q, is_equal
from nc_processes.arrival import Arrival
from nc_processes.constant_rate_server import ConstantRate
from nc_processes.service import Service


class Deconvolve(Arrival):
    """Deconvolution class."""

    def __init__(self, arr: Arrival, ser: Service, indep=True, p=1.0) -> None:
        self.arr = arr
        self.ser = ser

        if indep:
            self.p = 1.0
        else:
            self.p = p
        self.q = get_q(p=p, indep=indep)

    def sigma(self, theta: float) -> float:
        """

        :param theta: mgf parameter
        :return:      sigma(theta)
        """
        p_theta = self.p * theta
        q_theta = self.q * theta

        k_sig = -log(1 - exp(
            theta * (self.arr.rho(p_theta) - self.ser.rho(q_theta)))) / theta

        return self.arr.sigma(p_theta) + self.ser.sigma(q_theta) + k_sig

    def rho(self, theta: float) -> float:
        """

        :param theta: mgf parameter
        :return: rho(theta)
        """
        p_theta = self.p * theta
        q_theta = self.q * theta

        if self.arr.rho(p_theta) < 0 or self.ser.rho(q_theta) < 0:
            raise ParameterOutOfBounds("The rhos must be >= 0")

        if self.arr.rho(p_theta) >= self.ser.rho(q_theta):
            raise ParameterOutOfBounds(
                "The arrivals' rho has to be smaller than the service's rho")

        return self.arr.rho(p_theta)


class Convolve(Service):
    """Convolution class."""

    def __init__(self, ser1: Service, ser2: Service, indep=True,
                 p=1.0) -> None:
        self.ser1 = ser1
        self.ser2 = ser2
        if indep:
            self.p = 1.0
        else:
            self.p = p
        self.q = get_q(p=p, indep=indep)

    def sigma(self, theta: float) -> float:
        if isinstance(self.ser1, ConstantRate) and isinstance(
                self.ser2, ConstantRate):
            return 0.0

        p_theta = self.p * theta
        q_theta = self.q * theta

        if not is_equal(
                abs(self.ser1.rho(p_theta)), abs(self.ser2.rho(q_theta))):
            k_sig = -log(1 - exp(-theta * abs(
                self.ser1.rho(p_theta) - self.ser2.rho(q_theta)))) / theta

            return self.ser1.sigma(p_theta) + self.ser2.sigma(q_theta) + k_sig

        else:
            return self.ser1.sigma(p_theta) + self.ser2.sigma(q_theta)

    def rho(self, theta: float) -> float:
        if isinstance(self.ser1, ConstantRate) and isinstance(
                self.ser2, ConstantRate):
            return min(self.ser1.rate, self.ser2.rate)

        p_theta = self.p * theta
        q_theta = self.q * theta

        if self.ser1.rho(p_theta) < 0 or self.ser2.rho(q_theta) < 0:
            raise ParameterOutOfBounds("The rhos must be > 0")

        if not is_equal(
                abs(self.ser1.rho(p_theta)), abs(self.ser2.rho(q_theta))):
            return min(self.ser1.rho(p_theta), self.ser2.rho(q_theta))

        else:
            return self.ser1.rho(p_theta) - (1 / theta)


class Leftover(Service):
    """Subtract cross flow = nc_operations.Leftover class."""

    def __init__(self, arr: Arrival, ser: Service, indep=True, p=1.0) -> None:
        self.arr = arr
        self.ser = ser

        if indep:
            self.p = 1.0
        else:
            self.p = p
        self.q = get_q(p=p, indep=indep)

    def sigma(self, theta):
        p_theta = self.p * theta
        q_theta = self.q * theta

        return self.ser.sigma(q_theta) + self.arr.sigma(p_theta)

    def rho(self, theta):
        p_theta = self.p * theta
        q_theta = self.q * theta

        if self.ser.rho(q_theta) < 0 or self.arr.rho(p_theta) < 0:
            raise ParameterOutOfBounds("The rhos must be > 0")

        return self.ser.rho(q_theta) - self.arr.rho(p_theta)


class AggregateList(Arrival):
    """Multiple (list) aggregation class."""

    def __init__(self,
                 arr_list: List[Arrival],
                 p_list: List[float],
                 indep=True) -> None:
        self.arr_list = arr_list
        if indep:
            self.p_list = [1.0] * len(self.arr_list)
        else:
            if len(p_list) != (len(self.arr_list) - 1):
                raise ValueError(
                    f"number of p {len(p_list)} and length of "
                    f"arr_list {len(self.arr_list)} - 1 have to match")

            self.p_list = p_list.append(get_p_n(p_list=p_list, indep=indep))

    def sigma(self, theta: float) -> float:
        res = 0.0
        for i in range(len(self.arr_list)):
            res += self.arr_list[i].sigma(self.p_list[i] * theta)

        return res

    def rho(self, theta: float) -> float:
        for i in range(len(self.arr_list)):
            if self.arr_list[i].rho(self.p_list[i] * theta) < 0:
                raise ParameterOutOfBounds("The rhos must be > 0")

        res = 0.0
        for i in range(len(self.arr_list)):
            res += self.arr_list[i].rho(self.p_list[i] * theta)

        return res
