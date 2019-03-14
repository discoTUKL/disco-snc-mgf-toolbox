"""Implements all network operations in the sigma-rho calculus."""

from math import exp, log
from typing import List

from nc_arrivals.arrival import Arrival
from nc_operations.stability_check import stability_check
from nc_service.constant_rate_server import ConstantRate
from nc_service.service import Service
from utils.exceptions import ParameterOutOfBounds
from utils.helper_functions import get_p_n, get_q, is_equal

DELTA = 1e-05


class Deconvolve(Arrival):
    """Deconvolution class."""

    def __init__(self, arr: Arrival, ser: Service, indep=True, p=1.0) -> None:
        self.arr = arr
        self.ser = ser
        self.indep = indep

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

        if self.arr.is_discrete():
            return self.arr.sigma(p_theta) + self.ser.sigma(q_theta) + k_sig
        else:
            return self.arr.sigma(p_theta) + self.ser.sigma(
                q_theta) + self.arr.rho(p_theta) + k_sig

    def rho(self, theta: float) -> float:
        """

        :param theta: mgf parameter
        :return: rho(theta)
        """
        p_theta = self.p * theta
        q_theta = self.q * theta

        if self.arr.rho(p_theta) < 0 or self.ser.rho(q_theta) < 0:
            raise ParameterOutOfBounds("The rhos must be >= 0")

        stability_check(
            arr=self.arr,
            ser=self.ser,
            theta=theta,
            indep=self.indep,
            p=self.p)

        return self.arr.rho(p_theta)

    def is_discrete(self):
        return self.arr.is_discrete()


class Convolve(Service):
    """Convolution class."""

    def __init__(self, ser1: Service, ser2: Service, indep=True,
                 p=1.0) -> None:
        self.ser1 = ser1
        self.ser2 = ser2
        self.indep = indep

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

        if not is_equal(self.ser1.rho(p_theta), self.ser2.rho(q_theta)):
            k_sig = -log(1 - exp(-theta * abs(
                self.ser1.rho(p_theta) - self.ser2.rho(q_theta)))) / theta

            return self.ser1.sigma(p_theta) + self.ser2.sigma(q_theta) + k_sig

        else:
            if DELTA < 1 / theta:
                # if the DELTA rate reduction is smaller than the standard
                # approach
                return self.ser1.sigma(p_theta) + self.ser2.sigma(
                    q_theta) - log(1 - exp(-theta * DELTA))
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

        if not is_equal(self.ser1.rho(p_theta), self.ser2.rho(q_theta)):
            return min(self.ser1.rho(p_theta), self.ser2.rho(q_theta))

        else:
            return self.ser1.rho(p_theta) - min(1 / theta, DELTA)


class Leftover(Service):
    """Subtract cross flow = nc_operations.Leftover class."""

    def __init__(self, ser: Service, arr: Arrival, indep=True, p=1.0) -> None:
        self.arr = arr
        self.ser = ser
        self.indep = indep

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
            raise ParameterOutOfBounds("The rhos must be >= 0")

        return self.ser.rho(q_theta) - self.arr.rho(p_theta)


class AggregateList(Arrival):
    """Multiple (list) aggregation class."""

    def __init__(self,
                 arr_list: List[Arrival],
                 p_list: List[float],
                 indep=True) -> None:
        self.arr_list = arr_list
        if indep:
            self.p_list = [1.0]
        else:
            if len(p_list) != (len(self.arr_list) - 1):
                raise ValueError(
                    f"number of p {len(p_list)} and length of "
                    f"arr_list {len(self.arr_list)} - 1 have to match")

            self.p_list = p_list.append(get_p_n(p_list=p_list, indep=indep))
        self.indep = indep

    def sigma(self, theta: float) -> float:
        res = 0.0
        if self.indep:
            for i in range(len(self.arr_list)):
                res += self.arr_list[i].sigma(theta)
        else:
            for i in range(len(self.arr_list)):
                res += self.arr_list[i].sigma(self.p_list[i] * theta)

        return res

    def rho(self, theta: float) -> float:
        if self.indep:
            for i in range(len(self.arr_list)):
                if self.arr_list[i].rho(theta) < 0:
                    raise ParameterOutOfBounds("The rhos must be >= 0")
        else:
            for i in range(len(self.arr_list)):
                if self.arr_list[i].rho(self.p_list[i] * theta) < 0:
                    raise ParameterOutOfBounds("The rhos must be >= 0")

        res = 0.0
        if self.indep:
            for i in range(len(self.arr_list)):
                res += self.arr_list[i].rho(theta)
        else:
            for i in range(len(self.arr_list)):
                res += self.arr_list[i].rho(self.p_list[i] * theta)

        return res

    def is_discrete(self):
        return self.arr_list[0].is_discrete()
