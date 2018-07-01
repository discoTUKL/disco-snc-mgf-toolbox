"""Gather all distributions and parameter of the Monte Carlo simulation
in one class"""

from typing import List

from library.mc_enum import MCEnum


class MonteCarloDist(object):
    """Monte Carlo distribution class"""

    def __init__(self, mc_enum: MCEnum, param_list: List[float]) -> None:
        self.mc_enum = mc_enum
        self.param_list = param_list

    def to_name(self) -> str:
        return self.mc_enum.name

    def param_to_string(self) -> str:
        res = ""

        for i in range(len(self.param_list)):
            res += "parameter" + str(i + 1) + "_" + str(self.param_list[i])

        return res
