"""Implements the leftover process under GPS."""

from typing import List

from nc_arrivals.arrival import Arrival
from nc_server.server import Server
from utils.exceptions import IllegalArgumentError, ParameterOutOfBounds


class LeftoverGPSPG(Server):
    """Class to compute the leftover service for GPS."""
    def __init__(self, ser: Server, arr_list: List[Arrival],
                 phi_list: List[float]) -> None:
        self.ser = ser
        self.arr_list = arr_list

        if len(arr_list) != len(phi_list):
            raise IllegalArgumentError(f"len(arr_list) and len(phi_list) have "
                                       f"to match")

        self.phi_foi_weight = phi_list[0] / sum(phi_list)

    def sigma(self, theta):
        return self.phi_foi_weight * self.ser.sigma(theta=self.phi_foi_weight *
                                                    theta)

    def rho(self, theta):
        if self.ser.rho(theta=theta) < 0:
            raise ParameterOutOfBounds("The rhos must be >= 0")

        return self.phi_foi_weight * self.ser.rho(theta=self.phi_foi_weight *
                                                  theta)
