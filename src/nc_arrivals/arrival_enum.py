"""Enum class for arrival processes"""

from enum import Enum


class ArrivalEnum(Enum):
    """All implemented arrival processes"""
    DM1 = "D/M/1 queue"
    MD1 = "M/D/1 queue"
    MM1 = "M/M/1 queue"
    MMOOFluid = "Fluid Markov Modulated On-Off Traffic"
    MMOODisc = "Discrete Markov Modulated On-Off Traffic"
    EBB = "Exponentially Bounded Burstiness"
    TBConst = "Token-Bucket with constant parameters"
    MassOne = "Leaky Bucket following Massoulie"

    def number_parameters(self) -> int:
        if self == ArrivalEnum.DM1:
            return 1

        elif self == ArrivalEnum.MD1:
            return 1

        elif self == ArrivalEnum.MM1:
            return 1

        elif self == ArrivalEnum.MMOOFluid:
            return 3

        elif self == ArrivalEnum.MMOODisc:
            return 3

        elif self == ArrivalEnum.EBB:
            return 3

        elif self.TBConst:
            return 2

        elif self.MassOne:
            return 2

        else:
            raise NameError(f"Arrival process {self} is not implemented")
