"""Takes the Monte Carlo Enum and returns the random vector"""

import numpy as np

from library.mc_enum import MCEnum
from library.monte_carlo_dist import MonteCarloDist


def mc_enum_to_dist(mc_dist: MonteCarloDist, size: (int, int)) -> np.ndarray:
    if mc_dist.mc_enum == MCEnum.UNIFORM:
        return np.random.uniform(low=0, high=mc_dist.param_list[0], size=size)
    elif mc_dist.mc_enum == MCEnum.EXPONENTIAL:
        return np.random.exponential(scale=mc_dist.param_list[0], size=size)
    elif mc_dist.mc_enum == MCEnum.PARETO:
        return np.random.pareto(a=mc_dist.param_list[0], size=size)
    elif mc_dist.mc_enum == MCEnum.LOG_NORMAL:
        return np.random.lognormal(
            mean=mc_dist.param_list[0], sigma=mc_dist.param_list[1], size=size)
    elif mc_dist.mc_enum == MCEnum.CHI_SQUARED:
        return np.random.chisquare(df=mc_dist.param_list[0], size=size)
    else:
        raise NameError("Distribution parameter {0} is infeasible".format(
            mc_dist.mc_enum))
