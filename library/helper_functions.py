"""Helper functions"""

import functools
import warnings
from itertools import product
from typing import List

import numpy as np
import pandas as pd

EPSILON = 1e-09


def is_equal(float1, float2):
    """
    :param float1: real 1
    :param float2: real 2
    :return: returns true if distance is less than epsilon
    """
    return abs(float1 - float2) < EPSILON


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def expand_grid(list_input: list):
    """
    implement R-expand.grid() function
    :param list_input: list of values to be expanded
    :return:           expanded data frame
    """
    return pd.DataFrame([row for row in product(*list_input)])


def seq(start: float, stop: float, step: float) -> List[float]:
    """Implement the R-function in python."""

    n = int(round((stop - start) / step))
    #     The python round function rounds to next even
    #     number if equally distant

    if n > 1:
        return [start + step * i for i in range(n + 1)]
    else:
        return []


def opt_improvement_col(ar: np.array, metric: str):
    if metric == "relative":
        improvement_column = np.divide(ar[:, 0], ar[:, 1])
    elif metric == "absolute":
        improvement_column = np.subtract(ar[:, 0], ar[:, 1])
    else:
        raise NameError("Metric parameter {0} is infeasible".format(metric))

    return np.nanargmax(improvement_column)


def centroid_without_one_row(simplex: np.ndarray, index) -> np.ndarray:
    # type hint does not work with int and np.ndarray[int]
    # column mean of simplex without a given row
    # (usually the one with the worst y-value)
    return np.delete(simplex, index, 0).mean(axis=0)


def average_towards_best_row(simplex: np.ndarray, best_index,
                             shrink_factor: float) -> np.ndarray:
    # type hint does not work with int and np.ndarray[int]
    index = 0
    for row in simplex:
        simplex[index] = simplex[best_index] + shrink_factor * (
                row - simplex[best_index])
        index += 1

    return simplex


def get_unit_vector(length: int, index: int) -> List[float]:
    """

    :param length: length of unit vector
    :param index:  index of 1.0 value
    :return:       unit vector with 1.0 at index and 0.0 else
    """
    res = [0.0] * length
    res[index] = 1.0

    return res


if __name__ == '__main__':
    # SIMPLEX_START_TEST = np.array([[0.1, 2.0], [1.0, 3.0], [2.0, 2.0]])
    # print(SIMPLEX_START_TEST)
    # print(centroid_without_one_row(simplex=SIMPLEX_START_TEST, index=0))
    # print(
    #     average_towards_best_row(
    #         SIMPLEX_START_TEST, best_index=0, shrink_factor=0.5))
    # print(opt_improvement_col(SIMPLEX_START_TEST, 0, 1, "relative"))

    print(seq(0.1, 0.4, 0.2))
