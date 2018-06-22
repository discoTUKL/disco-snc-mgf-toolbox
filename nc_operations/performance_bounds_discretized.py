"""Performance bounds for continuous Process (need discretization)"""

from math import exp, inf, log

from library.exceptions import ParameterOutOfBounds
from library.helper_functions import is_equal
from nc_operations.performance_bounds import PerformanceBounds
from nc_processes.arrival import Arrival
from nc_processes.service import Service


class BacklogProbDiscretized(PerformanceBounds):
    """nc_operations.Backlog bound Violation Probability class"""

    def __init__(self, arr: Arrival, ser: Service) -> None:
        self.arr = arr
        self.ser = ser

    def bound(self, theta: float, backlog: float, tau=1.0) -> float:
        """Implements stationary bound method"""

        if self.arr.rho(theta) >= -self.ser.rho(theta):
            raise ParameterOutOfBounds(
                "The arrivals' rho {0} has to be smaller than"
                "the service's rho {1}".format(
                    self.arr.rho(theta), -self.ser.rho(theta)))

        sigma_arr_ser = self.arr.sigma(theta) + self.ser.sigma(theta)
        rho_arr_ser = self.arr.rho(theta) + self.ser.rho(theta)

        return exp(theta *
                   (-backlog + self.arr.rho(theta) * tau + sigma_arr_ser)) / (
                       1 - exp(theta * tau * rho_arr_ser))


class BacklogDiscretized(PerformanceBounds):
    """nc_operations.Backlog bound class"""

    def __init__(self, arr: Arrival, ser: Service) -> None:
        self.arr = arr
        self.ser = ser

    def bound(self, theta: float, prob_b: float, tau=1.0) -> float:
        """Implements stationary bound method"""

        if self.arr.rho(theta) >= -self.ser.rho(theta):
            raise ParameterOutOfBounds(
                "The arrivals' rho {0} has to be smaller than"
                "the service's rho {1}".format(
                    self.arr.rho(theta), -self.ser.rho(theta)))

        sigma_arr_ser = self.arr.sigma(theta) + self.ser.sigma(theta)
        rho_arr_ser = self.arr.rho(theta) + self.ser.rho(theta)

        log_part = log(prob_b * (1 - exp(theta * tau * rho_arr_ser)))

        return tau * self.arr.rho(theta) + sigma_arr_ser - log_part / theta


class DelayProbDiscretized(PerformanceBounds):
    """nc_operations.Delay bound Violation Probability class"""

    def __init__(self, arr: Arrival, ser: Service) -> None:
        self.arr = arr
        self.ser = ser

    def bound(self, theta: float, delay: int, tau=1.0) -> float:
        """Implements stationary bound method"""

        if self.arr.rho(theta) >= -self.ser.rho(theta):
            raise ParameterOutOfBounds(
                "The arrivals' rho {0} has to be smaller than"
                "the service's rho {1}".format(
                    self.arr.rho(theta), -self.ser.rho(theta)))

        sigma_arr_ser = self.arr.sigma(theta) + self.ser.sigma(theta)
        rho_arr_ser = self.arr.rho(theta) + self.ser.rho(theta)

        return exp(theta * (self.arr.rho(theta) * tau + sigma_arr_ser +
                            self.ser.rho(theta) * delay)) / (
                                1 - exp(theta * tau * rho_arr_ser))


class DelayDiscretized(PerformanceBounds):
    """nc_operations.Delay bound class"""

    def __init__(self, arr: Arrival, ser: Service) -> None:
        self.arr = arr
        self.ser = ser

    def bound(self, theta: float, prob_d: float, tau=1.0) -> float:
        """Implements stationary bound method"""

        if self.arr.rho(theta) >= -self.ser.rho(theta):
            raise ParameterOutOfBounds(
                "The arrivals' rho {0} has to be smaller than"
                "the service's rho {1}".format(
                    self.arr.rho(theta), -self.ser.rho(theta)))

        sigma_arr_ser = self.arr.sigma(theta) + self.ser.sigma(theta)
        rho_arr_ser = self.arr.rho(theta) + self.ser.rho(theta)

        log_part = log(prob_d * (1 - exp(theta * tau * rho_arr_ser)))

        return (log_part / theta - (tau * self.arr.rho(theta) + sigma_arr_ser)
                ) / self.ser.rho(theta)


class OutputDiscretized(PerformanceBounds):
    """nc_operations.Output bound class"""

    def __init__(self, arr: Arrival, ser: Service) -> None:
        self.arr = arr
        self.ser = ser

    def bound(self, theta: float, delta_time: int) -> float:
        """Implements stationary bound method"""

        if self.arr.rho(theta) >= -self.ser.rho(theta):
            raise ParameterOutOfBounds(
                "The arrivals' rho {0} has to be smaller than"
                "the service's rho {1}".format(
                    self.arr.rho(theta), -self.ser.rho(theta)))

        sigma_arr_ser = self.arr.sigma(theta) + self.ser.sigma(theta)
        rho_arr_ser = self.arr.rho(theta) + self.ser.rho(theta)

        numerator = exp(
            theta * (self.arr.rho(theta) * (delta_time + 1) + sigma_arr_ser))

        denominator = 1 - exp(theta * rho_arr_ser)

        if is_equal(denominator, 0):
            return inf

        return numerator / denominator
