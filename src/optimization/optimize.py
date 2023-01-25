"""Optimize theta and all other parameters"""

import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.optimize

from optimization.nelder_mead_parameters import NelderMeadParameters
from optimization.optimization_result import OptimizationResult
from optimization.sim_anneal_param import SimAnnealParams
from utils.deprecated import deprecated
from utils.exceptions import IllegalArgumentError, ParameterOutOfBounds
from utils.helper_functions import average_towards_best_row, centroid_without_one_row, expand_grid
from utils.setting import Setting


class Optimize(object):
    """Optimize class"""

    def __init__(self, setting: Setting, number_param: int) -> None:
        self.setting = setting
        self.number_param = number_param

    def eval_except(self, param_list: List[float]) -> float:
        """
        Shortens the exception handling and case distinction in a small method.

        :param param_list: theta ond other parameters
        :return:           function to_value
        """
        try:
            return self.setting.standard_bound(param_list=param_list)
        except (FloatingPointError, OverflowError, ParameterOutOfBounds):
            return math.inf

    def grid_search(self, grid_bounds: List[Tuple[float, float]], delta: float) -> OptimizationResult:
        """
        Search optimal values along a grid in the parameter space.

        :param grid_bounds: list of tuples of lower and upper bounds
        :param delta:      granularity of the grid search
        :return:           optimized standard_bound
        """
        if len(grid_bounds) != self.number_param:
            raise IllegalArgumentError(f"Number of parameters = {len(grid_bounds)} " f"!= {self.number_param}")

        list_slices = [slice(0)] * len(grid_bounds)

        for i in range(len(grid_bounds)):
            list_slices[i] = slice(grid_bounds[i][0], grid_bounds[i][1], delta)

        np.seterr("raise")

        # grid_res = scipy.optimize.brute(func=self.eval_except,
        #                                 ranges=tuple(list_slices),
        #                                 full_output=True)

        try:
            grid_res = scipy.optimize.brute(func=self.eval_except, ranges=tuple(list_slices), full_output=True)
        except FloatingPointError:
            return OptimizationResult(opt_x=[0.0] * self.number_param, obj_value=inf, heuristic="grid_search")

        return OptimizationResult(opt_x=grid_res[0].tolist(), obj_value=grid_res[1], heuristic="grid_search")

    def pattern_search(self, start_list: List[float], delta=3.0, delta_min=0.01) -> OptimizationResult:
        """
        Optimization in Hooke and Jeeves.

        :param start_list: list of starting values
        :param delta:      initial step length
        :param delta_min:  final step length
        :return:           optimized standard_bound
        """

        if len(start_list) != self.number_param:
            raise IllegalArgumentError(f"Number of parameters {len(start_list)} is wrong, "
                                       f"should be {self.number_param} instead")

        optimum_current = self.eval_except(param_list=start_list)

        optimum_new = optimum_current

        param_list = start_list[:]
        param_new = param_list[:]

        while delta > delta_min:
            for index, value in enumerate(param_list):
                param_new[index] = value + delta
                candidate_plus = self.eval_except(param_list=param_new)

                param_new[index] = value - delta
                candidate_minus = self.eval_except(param_list=param_new)

                if candidate_plus < optimum_new:
                    param_new[index] = value + delta
                    optimum_new = candidate_plus
                elif candidate_minus < optimum_new:
                    param_new[index] = value - delta
                    optimum_new = candidate_minus

            if optimum_new < optimum_current:
                # i.e., exploration step was successful
                param_old = param_list[:]
                param_list = param_new[:]
                optimum_current = optimum_new
                for index in range(len(param_list)):
                    param_new[index] = 2 * param_list[index] - param_old[index]

                # try a pattern step
                candidate_new = self.eval_except(param_list=param_new)

                if candidate_new < optimum_current:
                    param_list = param_new[:]
                    optimum_current = optimum_new
            else:
                param_new = param_list[:]
                delta *= 0.5

        return OptimizationResult(opt_x=param_list, obj_value=optimum_new, heuristic="pattern_search")

    def nelder_mead(self, simplex: np.ndarray, sd_min=10**(-2)) -> OptimizationResult:
        """
        Nelder-Mead optimization from the sciPy package.

        :param simplex:     initial parameter simplex
        :param sd_min:      abort criterion (detect when the changes
                            become very small)
        :return:            optimized standard_bound
        """
        np.seterr("raise")
        try:
            nm_res = scipy.optimize.minimize(self.eval_except,
                                             x0=np.zeros(shape=(1, simplex.shape[1])),
                                             method='Nelder-Mead',
                                             options={
                                                 'initial_simplex': simplex,
                                                 'fatol': sd_min
                                             })

        except FloatingPointError:
            return OptimizationResult(opt_x=[0.0] * self.number_param, obj_value=inf, heuristic="nelder_mead")

        return OptimizationResult(opt_x=nm_res.x, obj_value=nm_res.fun, heuristic="nelder_mead")

    def basin_hopping(self, start_list: List[float]) -> OptimizationResult:
        """
        Basin Hopping optimization from the sciPy package.

        :param start_list:  initial guess
        :return:            optimized standard_bound
        """
        try:
            bh_res = scipy.optimize.basinhopping(func=self.eval_except, x0=start_list)

        except FloatingPointError:
            return OptimizationResult(opt_x=[0.0] * self.number_param, obj_value=inf, heuristic="basin_hopping")

        return OptimizationResult(opt_x=bh_res.x, obj_value=bh_res.fun, heuristic="basin_hopping")

    def diff_evolution(self, bound_list: List[tuple]) -> OptimizationResult:
        """
        Differential Evolution optimization from the sciPy package.

        :param bound_list: list of tuples of lower and upper bounds
        :return:           optimized standard_bound
        """
        np.seterr("raise")

        try:
            de_res = scipy.optimize.differential_evolution(func=self.eval_except, bounds=bound_list)

        except FloatingPointError:
            return OptimizationResult(opt_x=[0.0] * self.number_param, obj_value=inf, heuristic="diff_evolution")

        return OptimizationResult(opt_x=de_res.x, obj_value=de_res.fun, heuristic="diff_evolution")

    def dual_annealing(self, bound_list: List[Tuple[float, float]]) -> OptimizationResult:
        np.seterr("raise")

        try:
            dual_anneal_res = scipy.optimize.dual_annealing(func=self.eval_except, bounds=bound_list)

        except (FloatingPointError, ValueError):
            return OptimizationResult(opt_x=[0.0] * self.number_param, obj_value=inf, heuristic="dual_annealing")

        return OptimizationResult(opt_x=dual_anneal_res.x, obj_value=dual_anneal_res.fun, heuristic="dual_annealing")

    @deprecated
    def sim_annealing(self, start_list: List[float], sim_anneal_params: SimAnnealParams) -> OptimizationResult:
        """

        :param start_list:       initial parameter set
        :param sim_anneal_params:  object that contains all the simulated
                                 annealing-parameters and helper methods
        :return:                 optimized standard_bound
        """

        param_list = start_list[:]
        optimum_current = self.eval_except(param_list=param_list)

        param_best = param_list[:]
        optimum_best = optimum_current

        temperature = sim_anneal_params.temp_start
        rep_max = sim_anneal_params.rep_max
        search_radius = sim_anneal_params.search_radius

        objective_change = True

        while objective_change:
            objective_change = False
            random_numbers = np.random.uniform(size=rep_max)

            for iteration in range(rep_max):
                param_new = sim_anneal_params.search_feasible_neighbor(objective=self.eval_except,
                                                                       input_list=param_list,
                                                                       search_radius=search_radius)
                optimum_new = self.eval_except(param_list=param_new)

                if optimum_new < optimum_current:
                    optimum_current = optimum_new
                    param_list = param_new[:]
                    objective_change = True
                    if optimum_new < optimum_best:
                        param_best = param_new[:]
                        optimum_best = optimum_new
                else:
                    if math.exp((optimum_current - optimum_new) / temperature) > random_numbers[iteration]:
                        # even if we compute inf - inf, Python does not
                        # lead to an error
                        param_list = param_new[:]
                        optimum_current = optimum_new
                        # Due to the fact that objective_new > objective, we
                        # have exp(-c/temperature), which is decreasing
                        # in the temperature
                        objective_change = True

            temperature *= sim_anneal_params.cooling_factor

        return OptimizationResult(opt_x=param_best, obj_value=optimum_best, heuristic="sim_annealing")

    @deprecated
    def grid_search_old(self, bound_list: List[Tuple[float, float]], delta: float) -> OptimizationResult:
        """
        Search optimal values along a grid in the parameter space.

        :param bound_list: list of tuples of lower and upper bounds
        :param delta:      granularity of the grid search
        :return:           optimized standard_bound
        """
        # first = lower standard_bound
        # second = upper standard_bound

        param_list = [[]] * len(bound_list)

        for index, value in enumerate(bound_list):
            param_list[index] = np.arange(start=value[0], stop=value[1] + 10**(-10), step=delta).tolist()

        # each entry in the dictionary consists of lower and upper bounds

        param_grid_df: pd.DataFrame = expand_grid(list_input=param_list)

        number_values = param_grid_df.shape[0]

        y_opt = inf
        opt_row = 0

        for row in range(number_values):
            candidate_opt = self.eval_except(param_list=param_grid_df.values.tolist()[row])
            if candidate_opt < y_opt:
                y_opt = candidate_opt
                opt_row = row

        return OptimizationResult(opt_x=param_grid_df.iloc[opt_row].tolist(),
                                  obj_value=y_opt,
                                  heuristic="grid_search_old")

    @deprecated
    def nelder_mead_old(self, simplex: np.ndarray, nelder_mead_param: NelderMeadParameters,
                        sd_min=10**(-2)) -> OptimizationResult:
        """
        Nelder-Mead Optimization.

        :param simplex:            initial parameter simplex
        :param nelder_mead_param:  object that contains all the
                                   Nelder-Mead-parameters
        :param sd_min:             abort criterion (detect when the changes
                                   become very small)
        :return:                   optimized standard_bound
        """
        number_rows = simplex.shape[0]
        number_columns = simplex.shape[1]
        # number of rows is the number of points = number of columns + 1
        # number of columns is the number of parameters
        if number_rows is not number_columns + 1:
            raise IllegalArgumentError(f"array argument is not a simplex, rows: {number_rows},"
                                       f" columns: {number_columns}")

        reflection_alpha = nelder_mead_param.reflection_alpha
        expansion_gamma = nelder_mead_param.expansion_gamma
        contraction_beta = nelder_mead_param.contraction_beta
        shrink_gamma = nelder_mead_param.shrink_gamma

        # print("simplex_start", simplex)
        # print("centroid", centroid_without_one_row(simplex=simplex, index=0))

        y_value = np.empty(number_rows)

        index = 0
        # print(simplex_start)
        for row in simplex:
            # print("row = ", row)
            y_value[index] = self.eval_except(param_list=row)
            index += 1

        # print("y_value: ", y_value)
        best_index: int = np.nanargmin(y_value)
        # print("best index: ", best_index)

        while np.std(y_value, ddof=1) > sd_min:
            # print("simplex = ", simplex)
            # print("y_value = ", y_value)
            # print("standard_deviation of y = ", np.std(y_value, ddof=1))

            worst_index: int = np.argmax(y_value)
            best_index: int = np.argmin(y_value)

            # compute centroid without worst row
            centroid = centroid_without_one_row(simplex=simplex, index=worst_index)
            # print("centroid: ", centroid)

            # print("worst_point: ", simplex_start[worst_index])

            p_reflection = (1 + reflection_alpha) * centroid - reflection_alpha * simplex[worst_index]

            # print("p_reflection", p_reflection)
            y_p_reflection = self.eval_except(param_list=p_reflection)
            # print("y_p_reflection", y_p_reflection)

            second_worst_index = np.argsort(y_value)[-2]
            # print("second_worst_index", second_worst_index)

            if y_p_reflection < y_value[best_index]:
                p_expansion = expansion_gamma * p_reflection + (1 - expansion_gamma) * centroid
                y_p_expansion = self.eval_except(param_list=p_expansion)

                if y_p_expansion < y_value[best_index]:
                    simplex[worst_index] = p_expansion
                    y_value[worst_index] = y_p_expansion
                else:
                    simplex[worst_index] = p_reflection
                    y_value[worst_index] = y_p_reflection

            elif y_p_reflection > y_value[second_worst_index]:
                if y_p_reflection < y_value[worst_index]:
                    simplex[worst_index] = p_reflection
                    y_value[worst_index] = y_p_reflection

                p_contraction = contraction_beta * simplex[worst_index] + (1 - contraction_beta) * centroid
                y_p_contraction = self.eval_except(param_list=p_contraction)

                if y_p_contraction < y_value[worst_index]:
                    simplex[worst_index] = p_contraction
                    y_value[worst_index] = y_p_contraction

                else:
                    simplex = average_towards_best_row(simplex=simplex,
                                                       best_index=best_index,
                                                       shrink_factor=shrink_gamma)
                    index = 0
                    for row in simplex:
                        y_value[index] = self.eval_except(param_list=row)
                        index += 1

            else:
                simplex[worst_index] = p_reflection
                y_value[worst_index] = y_p_reflection

        return OptimizationResult(opt_x=simplex[best_index], obj_value=y_value[best_index], heuristic="nelder_mead_old")

    def bfgs(self, start_list: list) -> OptimizationResult:
        np.seterr("raise")

        try:
            bfgs_res = scipy.optimize.minimize(fun=self.eval_except, x0=np.array(start_list), method="BFGS")

        except FloatingPointError:
            return OptimizationResult(opt_x=[0.0] * self.number_param, obj_value=inf, heuristic="bfgs")

        return OptimizationResult(opt_x=bfgs_res.x, obj_value=bfgs_res.fun, heuristic="bfgs")
