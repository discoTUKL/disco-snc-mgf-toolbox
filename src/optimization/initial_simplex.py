"""Create simplex with random values on-the-fly"""

from typing import List

import numpy as np

from library.helper_functions import get_unit_vector


class InitialSimplex(object):
    def __init__(self, parameters_to_optimize: int) -> None:
        self.parameters_to_optimize = parameters_to_optimize

        self.number_rows = parameters_to_optimize + 1
        self.number_columns = parameters_to_optimize

    def uniform_dist(self, max_theta=3.0, max_l=4.0) -> np.ndarray:
        res = np.random.uniform(low=0.0, high=max_theta, size=self.number_rows)
        res = np.reshape(res, (-1, self.number_rows)).transpose()

        if self.parameters_to_optimize > 1:
            for _i in range(self.number_columns - 1):
                l_column = np.random.uniform(
                    low=1.0, high=max_l, size=self.number_rows)
                l_column = np.reshape(l_column,
                                      (-1, self.number_rows)).transpose()

                res = np.concatenate((res, l_column), axis=1)

        return res

    def gao_han(self, start_list: List[float], tau=0.05) -> np.ndarray:
        res = np.empty([self.number_rows, self.number_columns])
        res[0] = np.array(start_list)

        for _i in range(1, self.number_rows):
            res[_i] = start_list + tau * np.array(
                get_unit_vector(length=self.number_columns, index=_i - 1))

        return res


if __name__ == "__main__":
    ONLY_THETA = InitialSimplex(parameters_to_optimize=1).uniform_dist(
        max_theta=0.5, max_l=1.2)
    TWO = InitialSimplex(parameters_to_optimize=2).uniform_dist(
        max_theta=0.5, max_l=1.2)
    THREE = InitialSimplex(parameters_to_optimize=3).uniform_dist(
        max_theta=0.5, max_l=1.2)
    # print(ONLY_THETA)
    # print(TWO)
    # print(THREE)

    print(
        InitialSimplex(parameters_to_optimize=2).gao_han(
            start_list=[1.2, 2.2]))
