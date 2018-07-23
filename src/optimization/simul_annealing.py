"""Summarize all Simulated Annealing parameters in one class."""

from math import inf
from typing import Callable, List

import numpy as np


class SimulAnnealing(object):
    """Gather all simulated annealing attributes in one class."""

    def __init__(self,
                 rep_max: int = 20,
                 temp_start: float = 100.0,
                 cooling_factor: float = 0.95,
                 search_radius: float = 1.0) -> None:
        self.rep_max = rep_max
        self.temp_start = temp_start
        self.cooling_factor = cooling_factor
        self.search_radius = search_radius

    @staticmethod
    def change_param_random(input_list: List[float],
                            search_radius: float) -> List[float]:
        """
        Find new parameters inside of a given radius.

        :param input_list:      initial parameter set
        :param search_radius:   search radius
        :return:                changed parameters
        """

        rand_vector = np.random.uniform(
            low=-search_radius, high=search_radius, size=len(input_list))

        output_list = [0.0] * len(input_list)

        for index, value in enumerate(input_list):
            output_list[index] = value + rand_vector[index]

        return output_list

    def search_feasible_neighbor(self, objective: Callable, input_list: list,
                                 search_radius: float) -> list:
        """
        search for feasible neighbor.

        :param objective:       function to be optimized
        :param input_list:      initial parameter set
        :param search_radius:   search radius
        :return:                feasible neighbor parameter set
        """

        param_new = self.change_param_random(
            input_list=input_list, search_radius=search_radius)

        value = objective(param_new)

        while value == inf:
            param_new = self.change_param_random(
                input_list=input_list, search_radius=search_radius)
            value = objective(param_list=param_new)

        return param_new
