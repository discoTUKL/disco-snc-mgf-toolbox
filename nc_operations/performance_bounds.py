"""Implements performance bounds"""
# TODO: add discretized version for continuous distributions

from math import exp, log

from library.exceptions import ParameterOutOfBounds
from library.helper_functions import is_equal
from nc_processes.arrival import Arrival
from nc_processes.service import Service


class BacklogProb(object):
    """nc_operations.Backlog bound Violation Probability class"""

    def __init__(self, arr: Arrival, ser: Service) -> None:
        self.arr = arr
        self.ser = ser

    def bound(self, theta: float, backlog: float) -> float:
        """Implements stationary bound method"""

        if self.arr.rho(theta) >= -self.ser.rho(theta):
            raise ParameterOutOfBounds(
                "The arrivals' rho {0} has to be smaller than"
                "the service's rho {1}".format(
                    self.arr.rho(theta), -self.ser.rho(theta)))

        sigma_arr_ser = self.arr.sigma(theta) + self.ser.sigma(theta)
        rho_arr_ser = self.arr.rho(theta) + self.ser.rho(theta)

        return exp(theta *
                   (-backlog + sigma_arr_ser)) / (1 - exp(theta * rho_arr_ser))

    def bound_t(self, theta: float, tt: int, backlog: float) -> float:
        """Implements time dependent method"""

        sigma_arr_ser = self.arr.sigma(theta) + self.ser.sigma(theta)
        rho_arr_ser = self.arr.rho(theta) + self.ser.rho(theta)

        if is_equal(self.arr.rho(theta), -self.ser.rho(theta)):
            return exp(theta * (-backlog + sigma_arr_ser)) * (tt + 1)

        elif self.arr.rho(theta) > -self.ser.rho(theta):
            return exp(theta *
                       (-backlog + rho_arr_ser * tt + sigma_arr_ser)) / (
                           1 - exp(-theta * rho_arr_ser))

        else:
            return self.bound(theta=theta, backlog=backlog)


class Backlog(object):
    """nc_operations.Backlog bound class"""

    def __init__(self, arr: Arrival, ser: Service) -> None:
        self.arr = arr
        self.ser = ser

    def bound(self, theta: float, prob_b: float) -> float:
        """Implements stationary bound method"""

        if self.arr.rho(theta) >= -self.ser.rho(theta):
            raise ParameterOutOfBounds(
                "The arrivals' rho {0} has to be smaller than"
                "the service's rho {1}".format(
                    self.arr.rho(theta), -self.ser.rho(theta)))

        sigma_arr_ser = self.arr.sigma(theta) + self.ser.sigma(theta)
        rho_arr_ser = self.arr.rho(theta) + self.ser.rho(theta)

        log_part = log(prob_b * (1 - exp(theta * rho_arr_ser)))

        return sigma_arr_ser - log_part / theta

    def bound_t(self, theta: float, prob_b: float, tt: int) -> float:
        """Implements time dependent method"""

        sigma_arr_ser = self.arr.sigma(theta) + self.ser.sigma(theta)
        rho_arr_ser = self.arr.rho(theta) + self.ser.rho(theta)

        if is_equal(self.arr.rho(theta), -self.ser.rho(theta)):
            log_part = log(prob_b / (tt + 1))

            return sigma_arr_ser - log_part / theta

        elif self.arr.rho(theta) > -self.ser.rho(theta):
            log_part = log(prob_b * (1 - exp(-theta * rho_arr_ser)))

            return rho_arr_ser * tt + sigma_arr_ser - log_part / theta

        else:
            return self.bound(theta=theta, prob_b=prob_b)


class DelayProb(object):
    """nc_operations.Delay bound Violation Probability class"""

    def __init__(self, arr: Arrival, ser: Service) -> None:
        self.arr = arr
        self.ser = ser

    def bound(self, theta: float, delay: int) -> float:
        """Implements stationary bound method"""

        if self.arr.rho(theta) >= -self.ser.rho(theta):
            raise ParameterOutOfBounds(
                "The arrivals' rho {0} has to be smaller than"
                "the service's rho {1}".format(
                    self.arr.rho(theta), -self.ser.rho(theta)))

        sigma_arr_ser = self.arr.sigma(theta) + self.ser.sigma(theta)
        rho_arr_ser = self.arr.rho(theta) + self.ser.rho(theta)

        return exp(theta * (sigma_arr_ser + self.ser.rho(theta) * delay)) / (
            1 - exp(theta * rho_arr_ser))

    def bound_t(self, theta: float, tt: int, delay: int) -> float:
        """Implements time dependent method"""

        sigma_arr_ser = self.arr.sigma(theta) + self.ser.sigma(theta)
        rho_arr_ser = self.arr.rho(theta) + self.ser.rho(theta)

        if is_equal(self.arr.rho(theta), -self.ser.rho(theta)):
            return exp(theta * (
                self.ser.rho(theta) * delay + sigma_arr_ser)) * (tt + 1)

        elif self.arr.rho(theta) > -self.ser.rho(theta):
            return exp(theta *
                       (self.arr.rho(theta) * tt + self.ser.rho(theta) *
                        (tt + delay) + sigma_arr_ser)) / (
                            1 - exp(-theta * rho_arr_ser))

        else:
            return self.bound(theta=theta, delay=delay)


class Delay(object):
    """nc_operations.Delay bound class"""

    def __init__(self, arr: Arrival, ser: Service) -> None:
        self.arr = arr
        self.ser = ser

    def bound(self, theta: float, prob_d: float) -> float:
        """Implements stationary bound method"""

        if self.arr.rho(theta) >= -self.ser.rho(theta):
            raise ParameterOutOfBounds(
                "The arrivals' rho {0} has to be smaller than"
                "the service's rho {1}".format(
                    self.arr.rho(theta), -self.ser.rho(theta)))

        sigma_arr_ser = self.arr.sigma(theta) + self.ser.sigma(theta)
        rho_arr_ser = self.arr.rho(theta) + self.ser.rho(theta)

        log_part = log(prob_d * (1 - exp(theta * rho_arr_ser)))

        return (log_part / theta - sigma_arr_ser) / self.ser.rho(theta)

    def bound_t(self, theta: float, prob_d: float, tt: int) -> float:
        """Implements time dependent method"""

        sigma_arr_ser = self.arr.sigma(theta) + self.ser.sigma(theta)
        rho_arr_ser = self.arr.rho(theta) + self.ser.rho(theta)

        if self.arr.rho(theta) == -self.ser.rho(theta):
            log_part = log(prob_d / (tt + 1))

            return (log_part / theta - sigma_arr_ser) / self.ser.rho(theta)

        elif self.arr.rho(theta) > -self.ser.rho(theta):
            log_part = log(prob_d * (1 - exp(-theta * rho_arr_ser)))

            return log_part / theta - (
                rho_arr_ser * tt + sigma_arr_ser) / self.ser.rho(theta)

        else:
            return self.bound(theta=theta, prob_d=prob_d)


class Output(object):
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

        return exp(theta *
                   (self.arr.rho(theta) * delta_time + sigma_arr_ser)) / (
                       1 - exp(theta * rho_arr_ser))

    def bound_t(self, theta: float, tt: int, ss: int) -> float:
        """Implements time dependent method"""

        sigma_arr_ser = self.arr.sigma(theta) + self.ser.sigma(theta)
        rho_arr_ser = self.arr.rho(theta) + self.ser.rho(theta)

        if is_equal(self.arr.rho(theta), -self.ser.rho(theta)):
            return exp(theta * (self.arr.rho(theta) *
                                (tt - ss) + sigma_arr_ser)) * (ss + 1)

        elif self.arr.rho(theta) > -self.ser.rho(theta):
            return exp(theta * (self.arr.rho(theta) * tt + self.ser.rho(
                theta) * ss + sigma_arr_ser) / 1 - exp(-theta * rho_arr_ser))

        else:
            return self.bound(theta=theta, delta_time=tt - ss)
