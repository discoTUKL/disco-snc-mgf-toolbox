"""All distribution parameter are collected in one object."""

from math import nan


class DistribParam(object):
    def __init__(self,
                 burst=nan,
                 lamb=nan,
                 mu=nan,
                 rate=nan,
                 rho=nan,
                 sigma=nan) -> None:
        self.burst = burst
        self.lamb = lamb
        self.mu = mu
        self.rate = rate
        self.rho = rho
        self.sigma = sigma

    def get_exp_string(self, number: int) -> str:
        return "_lamb" + str(number) + "_" + str(self.lamb)

    def get_service_string(self, number: int) -> str:
        return "_rate" + str(number) + "_" + str(self.rate)

    def get_mmoo_string(self, number: int) -> str:
        return "_mu" + str(number) + "_" + str(
            self.mu) + "_lamb" + str(number) + "_" + str(
                self.lamb) + "_burst" + str(number) + "_" + str(self.burst)

    def get_sigma_rho(self, number: int) -> str:
        return "_sigma" + str(number) + "_" + str(
            self.sigma) + "_rho" + str(number) + "_" + str(self.rho)
