"""Implements all network operations in the sigma-rho calculus."""

from math import exp, log
from typing import List

from nc_arrivals.arrival import Arrival
from nc_operations.stability_check import stability_check
from nc_server.constant_rate_server import ConstantRateServer
from nc_server.server import Server
from utils.exceptions import ParameterOutOfBounds
from utils.helper_functions import get_p_n, get_q, is_equal


class Deconvolve(Arrival):
    """Deconvolution class."""

    def __init__(self, arr: Arrival, ser: Server, indep=True, p=1.0) -> None:
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
        arr_sigma_p_theta = self.arr.sigma(self.p * theta)
        ser_sigma_q_theta = self.ser.sigma(self.q * theta)

        arr_rho_p_theta = self.arr.rho(self.p * theta)

        k_sig = -log(
            1 - exp(theta *
                    (arr_rho_p_theta - self.ser.rho(self.q * theta)))) / theta

        if self.arr.is_discrete():
            return arr_sigma_p_theta + ser_sigma_q_theta + k_sig
        else:
            return arr_sigma_p_theta + ser_sigma_q_theta + arr_rho_p_theta + \
                   k_sig

    def rho(self, theta: float) -> float:
        """

        :param theta: mgf parameter
        :return: rho(theta)
        """
        arr_rho_p_theta = self.arr.rho(self.p * theta)

        if arr_rho_p_theta < 0 or self.ser.rho(self.q * theta) < 0:
            raise ParameterOutOfBounds("The rhos must be >= 0")

        stability_check(arr=self.arr,
                        ser=self.ser,
                        theta=theta,
                        indep=self.indep,
                        p=self.p)

        return arr_rho_p_theta

    def is_discrete(self):
        return self.arr.is_discrete()


class Convolve(Server):
    """Convolution class."""

    def __init__(self,
                 ser1: Server,
                 ser2: Server,
                 indep=True,
                 p=1.0,
                 alter=False,
                 delta=1e-05) -> None:
        self.ser1 = ser1
        self.ser2 = ser2
        self.indep = indep

        if indep:
            self.p = 1.0
        else:
            self.p = p

        self.q = get_q(p=p, indep=indep)
        self.alter = alter
        self.delta = delta

    def sigma(self, theta: float) -> float:
        if isinstance(self.ser1, ConstantRateServer) and isinstance(
                self.ser2, ConstantRateServer):
            return 0.0

        ser_1_sigma_p_theta = self.ser1.sigma(self.p * theta)
        ser_2_sigma_q_theta = self.ser2.sigma(self.q * theta)

        ser_1_rho_p_theta = self.ser1.rho(self.p * theta)
        ser_2_rho_q_theta = self.ser2.rho(self.q * theta)

        if not is_equal(ser_1_rho_p_theta, ser_2_rho_q_theta):
            k_sig = -log(1 - exp(-theta * abs(ser_1_rho_p_theta -
                                              ser_2_rho_q_theta))) / theta

            return ser_1_sigma_p_theta + ser_2_sigma_q_theta + k_sig

        else:
            if self.alter:
                return ser_1_sigma_p_theta + ser_2_sigma_q_theta - log(
                    1 - exp(-theta * self.delta))

            return ser_1_sigma_p_theta + ser_2_sigma_q_theta

    def rho(self, theta: float) -> float:
        if isinstance(self.ser1, ConstantRateServer) and isinstance(
                self.ser2, ConstantRateServer):
            return min(self.ser1.rate, self.ser2.rate)

        ser_1_rho_p_theta = self.ser1.rho(self.p * theta)
        ser_2_rho_q_theta = self.ser2.rho(self.q * theta)

        if ser_1_rho_p_theta < 0 or ser_2_rho_q_theta < 0:
            raise ParameterOutOfBounds("The rhos must be > 0")

        if not is_equal(ser_1_rho_p_theta, ser_2_rho_q_theta):
            return min(ser_1_rho_p_theta, ser_2_rho_q_theta)

        else:
            if self.alter:
                return ser_1_rho_p_theta - self.delta

            return ser_1_rho_p_theta - 1 / theta


class Leftover(Server):
    """Subtract cross flow = nc_operations.Leftover class."""

    def __init__(self, ser: Server, arr: Arrival, indep=True, p=1.0) -> None:
        self.arr = arr
        self.ser = ser
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
                    f"number of p={len(p_list)} and length of "
                    f"arr_list={len(self.arr_list)} - 1 have to match")

            self.p_list = p_list
            self.p_n = get_p_n(p_list=p_list, indep=False)
        self.indep = indep

    def sigma(self, theta: float) -> float:
        res = 0.0
        if self.indep:
            for i in range(len(self.arr_list)):
                res += self.arr_list[i].sigma(theta)
        else:
            for i in range(len(self.arr_list) - 1):
                res += self.arr_list[i].sigma(self.p_list[i] * theta)

            res += self.arr_list[-1].sigma(self.p_n * theta)

        return res

    def rho(self, theta: float) -> float:
        res = 0.0

        if self.indep:
            for i in range(len(self.arr_list)):
                rho_i = self.arr_list[i].rho(theta)
                if rho_i < 0:
                    raise ParameterOutOfBounds("The rhos must be >= 0")

                res += rho_i

        else:
            for i in range(len(self.arr_list) - 1):
                rho_i = self.arr_list[i].rho(self.p_list[i] * theta)
                if rho_i < 0:
                    raise ParameterOutOfBounds("The rhos must be >= 0")

                res += rho_i

            rho_n = self.arr_list[-1].rho(self.p_n * theta)
            if rho_n < 0:
                raise ParameterOutOfBounds("The rhos must be >= 0")

            res += rho_n

        return res

    def is_discrete(self):
        return self.arr_list[0].is_discrete()


class AggregateTwo(Arrival):
    """Multiple (list) aggregation class."""

    def __init__(self, arr1: Arrival, arr2: Arrival, indep=True,
                 p=1.0) -> None:
        self.arr1 = arr1
        self.arr2 = arr2
        self.indep = indep
        if indep:
            self.p = 1.0
        else:
            self.p = p

        self.q = get_q(p=p, indep=indep)

    def sigma(self, theta: float) -> float:
        return self.arr1.sigma(self.p * theta) + self.arr2.sigma(
            self.q * theta)

    def rho(self, theta: float) -> float:
        arr_1_rho_p_theta = self.arr1.rho(self.p * theta)
        arr_2_rho_q_theta = self.arr2.rho(self.q * theta)

        if arr_1_rho_p_theta < 0 or arr_2_rho_q_theta < 0:
            raise ParameterOutOfBounds("The rhos must be >= 0")

        return arr_1_rho_p_theta + arr_2_rho_q_theta

    def is_discrete(self):
        return self.arr1.is_discrete()


if __name__ == '__main__':
    from timeit import default_timer as timer
    from nc_arrivals.regulated_arrivals import TokenBucketConstant
    ar_list = [
        TokenBucketConstant(sigma_single=1.0, rho_single=1.5, n=8),
        TokenBucketConstant(sigma_single=2.0, rho_single=3.0, n=10)
    ]

    start = timer()
    agg_list = AggregateList(arr_list=ar_list, p_list=[], indep=True)
    stop = timer()
    time_list = stop - start
    start = timer()
    agg_two = AggregateTwo(arr1=ar_list[0], arr2=ar_list[1], indep=True)
    stop = timer()
    time_two = stop - start

    print(f"sum sigma list = {agg_list.sigma(theta=1.0)}")
    print(f"sum rho list = {agg_list.rho(theta=1.0)}")
    print(f"time sigma two = {agg_two.sigma(theta=1.0)}")
    print(f"time rho two = {agg_two.rho(theta=1.0)}")

    print(f"time list = {time_list} s")
    print(f"time two = {time_two} s")
