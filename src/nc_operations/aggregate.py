"""Aggregation classes."""

from typing import List

import numpy as np
from nc_arrivals.arrival import Arrival
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.regulated_arrivals import DetermTokenBucket
from utils.exceptions import IllegalArgumentError, ParameterOutOfBounds
from utils.helper_functions import get_p_n, get_q


class AggregateList(Arrival):
    """Multiple (list) aggregation class."""
    def __init__(self, arr_list: List[Arrival], indep: bool,
                 p_list: List[float]) -> None:
        self.arr_list = arr_list[:]
        if indep:
            self.p_list = [1.0]
            # self.p_n = 1.0
        else:
            if len(p_list) != (len(self.arr_list) - 1):
                raise IllegalArgumentError(
                    f"number of p={len(p_list)} and length of "
                    f"arr_list={len(self.arr_list)} - 1 have to match")

            self.p_list = p_list[:]
            if isinstance(p_list, np.ndarray):
                np.concatenate((self.p_list, get_p_n(p_list=p_list)))
            else:
                self.p_list.append(get_p_n(p_list=p_list))
        self.indep = indep

    def sigma(self, theta: float) -> float:
        res = 0.0
        if self.indep:
            for arr in self.arr_list:
                res += arr.sigma(theta)
        else:
            for i, arr in enumerate(self.arr_list):
                res += arr.sigma(self.p_list[i] * theta)

        return res

    def rho(self, theta: float) -> float:
        res = 0.0

        if self.indep:
            for arrival in self.arr_list:
                rho_i = arrival.rho(theta)
                if rho_i < 0:
                    raise ParameterOutOfBounds("The rhos must be >= 0")

                res += rho_i

        else:
            for i, arr in enumerate(self.arr_list):
                rho_i = arr.rho(self.p_list[i] * theta)
                if rho_i < 0:
                    raise ParameterOutOfBounds("The rhos must be >= 0")

                res += rho_i

        return res

    def is_discrete(self):
        return self.arr_list[0].is_discrete()


class AggregateTwo(Arrival):
    """Multiple (list) aggregation class."""
    def __init__(self,
                 arr1: Arrival,
                 arr2: Arrival,
                 indep=True,
                 p=1.0) -> None:
        self.arr1 = arr1
        self.arr2 = arr2
        self.indep = indep
        if indep:
            self.p = 1.0
            self.q = 1.0
        else:
            self.p = p
            self.q = get_q(p=p)

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


class AggregateHomogeneous(Arrival):
    """Multiple (list) aggregation class."""
    def __init__(self, arr: Arrival, n: int, indep=True) -> None:
        self.arr = arr
        self.n = n

        if not indep:
            raise NotImplementedError(
                "only the independet case is implemented")

    def sigma(self, theta: float) -> float:
        return self.n * self.arr.sigma(theta=theta)

    def rho(self, theta: float) -> float:
        return self.n * self.arr.rho(theta=theta)

    def is_discrete(self):
        return self.arr.is_discrete()

    def average_rate(self) -> float:
        if isinstance(self.arr, ArrivalDistribution):
            return self.n * self.arr.average_rate()


if __name__ == '__main__':
    from timeit import default_timer as timer

    ARR_LIST = [
        DetermTokenBucket(sigma_single=1.0, rho_single=1.5, m=8),
        DetermTokenBucket(sigma_single=2.0, rho_single=3.0, m=10)
    ]

    START = timer()
    AGG_LIST = AggregateList(arr_list=ARR_LIST, indep=False, p_list=[2.0])
    STOP = timer()
    TIME_LIST = STOP - START
    START = timer()
    AGG_TWO = AggregateTwo(arr1=ARR_LIST[0],
                           arr2=ARR_LIST[1],
                           indep=False,
                           p=2.0)
    STOP = timer()
    TIME_TWO = STOP - START

    print(f"sum sigma list = {AGG_LIST.sigma(theta=1.0)}")
    print(f"time sigma two = {AGG_TWO.sigma(theta=1.0)}")
    print(f"sum rho list = {AGG_LIST.rho(theta=1.0)}")
    print(f"time rho two = {AGG_TWO.rho(theta=1.0)}")

    print(f"time list = {TIME_LIST} s")
    print(f"time two = {TIME_TWO} s")
