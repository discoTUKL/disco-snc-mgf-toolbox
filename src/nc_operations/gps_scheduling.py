"""Implements the leftover process under GPS."""

from typing import List

from nc_server.server import Server
from utils.exceptions import ParameterOutOfBounds


class LeftoverGPSPG(Server):
    """Class to compute the leftover service for GPS."""

    def __init__(self, ser: Server, phi_list: List[float]) -> None:
        self.ser = ser
        self.phi_foi_weight = phi_list[0] / sum(phi_list)

    def sigma(self, theta):
        return self.phi_foi_weight * self.ser.sigma(theta=self.phi_foi_weight * theta)

    def rho(self, theta):
        if self.ser.rho(theta=theta) < 0:
            raise ParameterOutOfBounds("The rhos must be >= 0")

        return self.phi_foi_weight * self.ser.rho(theta=self.phi_foi_weight * theta)
