"""Helper function to manipulate data"""

import numpy as np


def remove_nan_rows(full_array: np.array) -> np.array:
    return full_array[~np.isnan(full_array).any(axis=1)]
