"""Takes the Monte Carlo Enum and returns the random vector"""

import numpy as np

from bound_evaluation.mc_enum import MCEnum
from bound_evaluation.monte_carlo_dist import MonteCarloDist
from nc_arrivals.arrival_enum import ArrivalEnum


def mc_enum_to_dist(arrival_enum: ArrivalEnum, mc_dist: MonteCarloDist,
                    number_flows: int, number_servers: int,
                    total_iterations: int) -> np.ndarray:
    size_array = [
        total_iterations,
        arrival_enum.number_parameters() * number_flows + number_servers
        # servers have only 1 parameter
    ]

    if arrival_enum == ArrivalEnum.MMOODisc:
        probabilities = np.random.uniform(
            low=0.0, high=1.0, size=[total_iterations, 2 * number_flows])
        if mc_dist.mc_enum == MCEnum.UNIFORM:
            return np.concatenate(
                (probabilities,
                 np.random.uniform(
                     low=0.0,
                     high=mc_dist.param_list[0],
                     size=[total_iterations, number_flows + number_servers])),
                axis=1)
        elif mc_dist.mc_enum == MCEnum.EXPONENTIAL:
            return np.concatenate(
                (probabilities,
                 np.random.exponential(
                     scale=1 / mc_dist.param_list[0],
                     size=[total_iterations, number_flows + number_servers])),
                axis=1)
        # watch out: scale is the expectation 1 / lambda
        elif mc_dist.mc_enum == MCEnum.PARETO:
            return np.concatenate(
                (probabilities,
                 np.random.pareto(
                     a=mc_dist.param_list[0],
                     size=[total_iterations, number_flows + number_servers])),
                axis=1)
        elif mc_dist.mc_enum == MCEnum.LOG_NORMAL:
            return np.concatenate(
                (probabilities,
                 np.random.lognormal(
                     mean=mc_dist.param_list[0],
                     sigma=mc_dist.param_list[1],
                     size=[total_iterations, number_flows + number_servers])),
                axis=1)
        elif mc_dist.mc_enum == MCEnum.CHI_SQUARED:
            return np.concatenate(
                (probabilities,
                 np.random.chisquare(
                     df=mc_dist.param_list[0],
                     size=[total_iterations, number_flows + number_servers])),
                axis=1)
        else:
            raise NameError(
                f"Distribution parameter {mc_dist.mc_enum} is infeasible")

    else:
        if mc_dist.mc_enum == MCEnum.UNIFORM:
            return np.random.uniform(low=0.0,
                                     high=mc_dist.param_list[0],
                                     size=size_array)
        elif mc_dist.mc_enum == MCEnum.EXPONENTIAL:
            return np.random.exponential(scale=1 / mc_dist.param_list[0],
                                         size=size_array)
        # watch out: scale is the expectation 1 / lambda
        elif mc_dist.mc_enum == MCEnum.PARETO:
            return np.random.pareto(a=mc_dist.param_list[0], size=size_array)
        elif mc_dist.mc_enum == MCEnum.LOG_NORMAL:
            return np.random.lognormal(mean=mc_dist.param_list[0],
                                       sigma=mc_dist.param_list[1],
                                       size=size_array)
        elif mc_dist.mc_enum == MCEnum.CHI_SQUARED:
            return np.random.chisquare(df=mc_dist.param_list[0],
                                       size=size_array)
        else:
            raise NameError(
                f"Distribution parameter {mc_dist.mc_enum} is infeasible")
