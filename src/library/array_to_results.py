"""This file takes arrays and writes them into dictionaries"""

import numpy as np
from warnings import warn

from nc_processes.arrival_distribution import (MMOO, ArrivalDistribution,
                                               ExponentialArrival)
from nc_processes.constant_rate_server import ConstantRate


def data_array_to_results(arrival: ArrivalDistribution,
                          const_rate: ConstantRate,
                          param_array: np.array,
                          res_array: np.array,
                          number_servers: int,
                          metric="relative") -> dict:
    """Writes the array values into a dictionary, MMOO"""

    if metric == "relative":
        improvement_vec = np.divide(res_array[:, 0], res_array[:, 1])
    elif metric == "absolute":
        improvement_vec = np.subtract(res_array[:, 0], res_array[:, 1])
    else:
        raise NameError("Metric parameter {0} is infeasible".format(metric))

    row_max = np.nanargmax(improvement_vec)
    opt_standard_bound = res_array[row_max, 0]
    opt_new_bound = res_array[row_max, 1]
    opt_improvement = improvement_vec[row_max]
    mean_improvement = np.nanmean(improvement_vec)

    count_nan = np.count_nonzero(~np.isnan(res_array), axis=0)
    count_nan_standard = count_nan[0]
    count_nan_new = count_nan[1]

    if count_nan_standard != count_nan_new:
        warn("number of nan's does not match, {0} != {1}".format(
            count_nan_standard, count_nan_new))

    if count_nan_standard > int(res_array.shape[0] / 10):
        warn("way to many nan's: {0} out of {1}!".format(
            count_nan_standard, int(res_array.shape[0])))

    res_dict = {"Name": "Value", "arrival_distribution": arrival.to_name()}

    for j in range(number_servers):
        if isinstance(arrival, ExponentialArrival):
            res_dict["lamb{0}".format(j + 1)] = param_array[row_max, j]
            res_dict["rate{0}".format(j + 1)] = param_array[row_max,
                                                            number_servers + j]

        elif isinstance(arrival, MMOO):
            res_dict["mu{0}".format(j + 1)] = param_array[row_max, j]
            res_dict["lamb{0}".format(j + 1)] = param_array[row_max,
                                                            number_servers + j]
            res_dict["burst{0}".format(j + 1)] = param_array[
                row_max, 2 * number_servers + j]
            res_dict["rate{0}".format(j + 1)] = param_array[
                row_max, 3 * number_servers + j]

        else:
            raise NameError(
                "Arrival parameter {0}or Service parameter{1} is infeasible".
                format(arrival.to_name(), const_rate.to_name()))

    res_dict.update({
        "opt_standard_bound": opt_standard_bound,
        "opt_new_bound": opt_new_bound
    })
    if metric == "relative":
        res_dict.update({
            "optimum_improvement_factor": opt_improvement,
            "mean_improvement_factor": mean_improvement
        })
    elif metric == "absolute":
        res_dict.update({
            "optimum_absolute_improvement": opt_improvement,
            "mean_absolute_improvement": mean_improvement
        })
    else:
        raise NameError("Metric parameter {0} is infeasible".format(metric))

    # negative improvement cannot occur for grid_search_old
    # row_min = np.nanargmin(res_array, axis=0)[-1]

    # if res_array[row_min][-1] < 0:
    #     worst_dict = {}
    #     for j in range(number_servers):
    #         worst_dict["mu{0}".format(j + 1)] = res_array[row_min, j]
    #         worst_dict["lamb{0}".format(j + 1)] = res_array[
    #             row_min, number_servers + j]
    #         worst_dict["burst{0}".format(j + 1)] = res_array[
    #             row_min, 2 * number_servers + j]
    #         worst_dict["rate{0}".format(j + 1)] = res_array[
    #             row_min, 3 * number_servers + j]
    #
    #     # bound = [-3], new_bound = [-2], opt_diff_new = [-1]
    #     worst_dict.update({
    #         "bound": res_array[row_min, -3],
    #         "new_bound": res_array[row_min, -2],
    #         "abs/rel": res_array[row_min, -1]
    #     })
    #
    #     print(worst_dict)

    return res_dict


def time_array_to_results(arrival: ArrivalDistribution, time_array,
                          number_servers: int, time_ratio: dict) -> dict:
    """Writes the array values into a dictionary"""

    standard_mean = np.nanmean(time_array[:, 0])
    lyapunov_mean = np.nanmean(time_array[:, 1])

    ratio = np.divide(time_array[:, 0], time_array[:, 1])
    mean_ratio = np.nanmean(ratio)

    time_ratio.update({number_servers: mean_ratio})

    # time_standard = [-2], time_lyapunov = [-1]
    res_dict = {
        "Name": "Value",
        "arrival_distribution": arrival.to_name(),
        "number_servers": number_servers,
        "standard_mean": standard_mean,
        "Lyapunov_mean": lyapunov_mean,
        "mean_fraction": mean_ratio
    }

    return res_dict
