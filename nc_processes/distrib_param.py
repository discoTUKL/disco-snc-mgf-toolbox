"""All distribution parameter are collected in one object."""

from math import nan


class DistribParam(object):
    def __init__(self, lamb=nan, mu=nan, burst=nan, rate=nan) -> None:
        self.lamb = lamb
        self.mu = mu
        self.burst = burst
        self.rate = rate

    def get_exp_string(self, number: int) -> str:
        return "_lamb" + str(number) + "_" + str(self.lamb)

    def get_mmoo_string(self, number: int) -> str:
        return "_mu" + str(number) + "_" + str(
            self.mu) + "_lamb" + str(number) + "_" + str(
                self.lamb) + "_burst" + str(number) + "_" + str(self.burst)

    def get_constant_string(self, number: int) -> str:
        return "_rate" + str(number) + "_" + str(self.rate)
