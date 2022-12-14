"""Helper functions"""

from itertools import product
from math import isinf
from typing import List

import mpmath as mp
import numpy as np
import pandas as pd

from utils.exceptions import ParameterOutOfBounds

EPSILON = 1e-09


def get_q(p: float) -> float:
    """
    :param p: Hoelder p
    :return: q
    """
    if p <= 1.0:
        raise ParameterOutOfBounds(f"p={p} must be >1")

    # 1/p + 1/q = 1
    # 1/q = (p - 1) / p

    return p / (p - 1.0)


def get_p_n(p_list) -> float:
    """
    :param p_list: first p_1, ..., p_n in generalized Hoelder inequality
    :return: last p_n
    """
    inv_p = [0.0] * len(p_list)
    for i in range(len(p_list)):
        if p_list[i] <= 1.0:
            raise ParameterOutOfBounds(f"p={p_list[i]} must be >1")
        inv_p[i] = 1.0 / p_list[i]
        if sum(inv_p) >= 1:
            raise ParameterOutOfBounds(f"p_i's are too small")

    return 1.0 / (1.0 - sum(inv_p))


def is_equal(float1: float, float2: float, epsilon=EPSILON) -> bool:
    """
    :param float1: real 1
    :param float2: real 2
    :param epsilon: accuracy of the comparison (default is the global Epsilon)
    :return: returns true if distance is less than epsilon
    """
    if isinf(float1) and isinf(float2):
        return True

    return abs(float1 - float2) < epsilon


def expand_grid(list_input: list) -> pd.DataFrame:
    """
    implement R-expand.grid() function
    :param list_input: list of values to be expanded
    :return:           expanded data frame
    """
    return pd.DataFrame([row for row in product(*list_input)])


def centroid_without_one_row(simplex: np.ndarray, index: int) -> np.ndarray:
    # type hint does not work with int and np.ndarray[int]
    # column mean of simplex without a given row
    # (usually the one with the worst y-to_value)
    return np.delete(simplex, index, 0).mean(axis=0)


def average_towards_best_row(simplex: np.ndarray, best_index: int,
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
    :param index:  index of 1.0 to_value
    :return:       unit vector with 1.0 at index and 0.0 else
    """
    res = [0.0] * length
    res[index] = 1.0

    return res


def same_sign(a, b):
    return a * b > 0


def bisect(func, low, high) -> float:
    """Find root of continuous function where f(low) and f(high) have
    opposite signs"""

    counter = 20

    while func(low) >= -EPSILON and counter > 0:
        low /= 2.0
        counter -= 1

    if func(low) >= 0:
        return low

    while func(high) <= EPSILON and counter > 0:
        high *= 2.0
        counter -= 1

    assert not same_sign(func(low), func(high))

    midpoint = mp.mpf((low + high) / 2.0)

    for i in range(54):
        midpoint = mp.mpf((low + high) / 2.0)
        if same_sign(func(low), func(midpoint)):
            low = mp.mpf(midpoint)
        else:
            high = mp.mpf(midpoint)
        if abs(high - low) < EPSILON:
            break

    return midpoint


if __name__ == '__main__':
    print(get_p_n(p_list=[3.0, 3.0]))

    SIMPLEX_START_TEST = np.array([[0.1, 2.0], [1.0, 3.0], [2.0, 2.0]])
    print(SIMPLEX_START_TEST)
    print(centroid_without_one_row(simplex=SIMPLEX_START_TEST, index=0))
    print(
        average_towards_best_row(SIMPLEX_START_TEST,
                                 best_index=0,
                                 shrink_factor=0.5))
