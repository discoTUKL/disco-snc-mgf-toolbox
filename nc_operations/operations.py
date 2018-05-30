"""Implements all network operations in the sigma-rho calculus."""

from math import exp, log
from typing import List

from library.helper_functions import is_equal
from library.exceptions import ParameterOutOfBounds
from nc_processes.arrival import Arrival
from nc_processes.service import Service


class Deconvolve(Arrival):
    """Deconvolution class."""

    def __init__(self, arr: Arrival, ser: Service) -> None:
        self.arr = arr
        self.ser = ser

    def sigma(self, theta: float) -> float:
        """

        :param theta: mgf parameter
        :return:      sigma(theta)
        """
        k_sig = -log(1 - exp(theta * (
            self.arr.rho(theta) + self.ser.rho(theta)))) / theta

        return self.arr.sigma(theta) + self.ser.sigma(theta) + k_sig

    def rho(self, theta: float) -> float:
        """

        :param theta: mgf parameter
        :return: rho(theta)
        """
        if self.arr.rho(theta) < 0 or self.ser.rho(theta) > 0:
            raise ParameterOutOfBounds("Check rho's sign")

        if self.arr.rho(theta) >= -self.ser.rho(theta):
            raise ParameterOutOfBounds(
                "The arrivals' rho has to be smaller than the service's rho")

        return self.arr.rho(theta)


class Convolve(Service):
    """Convolution class."""

    def __init__(self, ser1: Service, ser2: Service) -> None:
        self.ser1 = ser1
        self.ser2 = ser2

    def sigma(self, theta: float) -> float:
        if not is_equal(abs(self.ser1.rho(theta)), abs(self.ser2.rho(theta))):
            k_sig = -(1 / theta) * log(1 - exp(
                -theta * abs(self.ser1.rho(theta) - self.ser2.rho(theta))))

            return self.ser1.sigma(theta) + self.ser2.sigma(theta) + k_sig

        else:
            return self.ser1.sigma(theta) + self.ser2.sigma(theta)

    def rho(self, theta: float) -> float:
        if self.ser1.rho(theta) > 0 or self.ser2.rho(theta) > 0:
            raise ParameterOutOfBounds("Check rho's sign")

        if not is_equal(abs(self.ser1.rho(theta)), abs(self.ser2.rho(theta))):
            return max(self.ser1.rho(theta), self.ser2.rho(theta))

        else:
            return self.ser2.rho(theta) + (1 / theta)


class Leftover(Service):
    """Subtract cross flow = nc_operations.Leftover class."""

    def __init__(self, arr: Arrival, ser: Service) -> None:
        self.arr = arr
        self.ser = ser

    def sigma(self, theta):
        return self.arr.sigma(theta) + self.ser.sigma(theta)

    def rho(self, theta):
        if self.arr.rho(theta) < 0:
            print(self.arr.rho(theta))
            raise ParameterOutOfBounds("Arrivals' rho must be >= 0")

        if self.ser.rho(theta) > 0:
            print(self.ser.rho(theta))
            raise ParameterOutOfBounds("Service's rho must be <= 0")

        return self.arr.rho(theta) + self.ser.rho(theta)


class Aggregate(Arrival):
    """Single aggregation class"""

    def __init__(self, arr1: Arrival, arr2: Arrival) -> None:
        self.arr1 = arr1
        self.arr2 = arr2

    def sigma(self, theta: float) -> float:
        return self.arr1.sigma(theta) + self.arr2.sigma(theta)

    def rho(self, theta: float) -> float:
        if self.arr1.rho(theta) < 0 or self.arr2.rho(theta) < 0:
            raise ParameterOutOfBounds("Check rho's sign")

        return self.arr1.rho(theta) + self.arr2.rho(theta)


class AggregateList(Arrival):
    """Multiple (list) aggregation class."""

    def __init__(self, arr_list: List[Arrival]) -> None:
        self.arr_list = arr_list

    def sigma(self, theta: float) -> float:
        sigma_list = [x.sigma(theta) for x in self.arr_list]

        return sum(sigma_list)

    def rho(self, theta: float) -> float:
        # There is no sign checker implemented yet
        rho_list = [x.rho(theta) for x in self.arr_list]

        return sum(rho_list)
