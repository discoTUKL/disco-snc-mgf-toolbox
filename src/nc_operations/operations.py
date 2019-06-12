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
        p_theta = self.p * theta
        q_theta = self.q * theta

        k_sig = -log(
            1 - exp(theta *
                    (self.arr.rho(p_theta) - self.ser.rho(q_theta)))) / theta

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

        stability_check(arr=self.arr,
                        ser=self.ser,
                        theta=theta,
                        indep=self.indep,
                        p=self.p)

        return self.arr.rho(p_theta)

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

        p_theta = self.p * theta
        q_theta = self.q * theta

        if not is_equal(self.ser1.rho(p_theta), self.ser2.rho(q_theta)):
            k_sig = -log(1 - exp(-theta * abs(
                self.ser1.rho(p_theta) - self.ser2.rho(q_theta)))) / theta

            return self.ser1.sigma(p_theta) + self.ser2.sigma(q_theta) + k_sig

        else:
            if self.alter:
                return self.ser1.sigma(p_theta) + self.ser2.sigma(
                    q_theta) - log(1 - exp(-theta * self.delta))

            return self.ser1.sigma(p_theta) + self.ser2.sigma(q_theta)

    def rho(self, theta: float) -> float:
        if isinstance(self.ser1, ConstantRateServer) and isinstance(
                self.ser2, ConstantRateServer):
            return min(self.ser1.rate, self.ser2.rate)

        p_theta = self.p * theta
        q_theta = self.q * theta

        if self.ser1.rho(p_theta) < 0 or self.ser2.rho(q_theta) < 0:
            raise ParameterOutOfBounds("The rhos must be > 0")

        if not is_equal(self.ser1.rho(p_theta), self.ser2.rho(q_theta)):
            return min(self.ser1.rho(p_theta), self.ser2.rho(q_theta))

        else:
            if self.alter:
                return self.ser1.rho(p_theta) - self.delta

            return self.ser1.rho(p_theta) - 1 / theta


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

            p_list.append(get_p_n(p_list=p_list, indep=False))
            self.p_list = p_list
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
        res = 0.0

        if self.indep:
            for i in range(len(self.arr_list)):
                if self.arr_list[i].rho(theta) < 0:
                    raise ParameterOutOfBounds("The rhos must be >= 0")

                res += self.arr_list[i].rho(theta)

        else:
            for i in range(len(self.arr_list)):
                if self.arr_list[i].rho(self.p_list[i] * theta) < 0:
                    raise ParameterOutOfBounds("The rhos must be >= 0")

                res += self.arr_list[i].rho(self.p_list[i] * theta)

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
        p_theta = self.p * theta
        q_theta = self.q * theta

        return self.arr1.sigma(p_theta) + self.arr2.sigma(q_theta)

    def rho(self, theta: float) -> float:
        p_theta = self.p * theta
        q_theta = self.q * theta

        if self.arr1.rho(p_theta) < 0 or self.arr2.rho(q_theta) < 0:
            raise ParameterOutOfBounds("The rhos must be >= 0")

        return self.arr1.rho(p_theta) + self.arr2.rho(q_theta)

    def is_discrete(self):
        return self.arr1.is_discrete()


if __name__ == '__main__':
    from timeit import default_timer as timer
    from nc_arrivals.regulated_arrivals import TokenBucketConstant
    ar_list = [TokenBucketConstant(sigma_single=1.0, rho_single=1.5, n=8),
               TokenBucketConstant(sigma_single=2.0, rho_single=3.0, n=10)]

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
