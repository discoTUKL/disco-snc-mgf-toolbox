"""This file takes arrays and writes them into dictionaries"""

import numpy as np

from library.helper_functions import opt_improvement_col
from nc_processes.arrival_distribution import (MMOO, ArrivalDistribution,
                                               ExponentialArrival)
from nc_processes.service import ConstantRate, Service


def data_array_to_results(arrival: ArrivalDistribution,
                          service: Service,
                          param_array: np.array,
                          res_array: np.array,
                          number_servers: int,
                          metric="relative") -> dict:
    """Writes the array values into a dictionary, MMOO"""

    mean_standard_bound = np.nanmean(res_array[:, 0])
    mean_new_bound = np.nanmean(res_array[:, 1])

    row_max = opt_improvement_col(res_array, 0, 1, metric)
    opt_standard_bound = res_array[row_max, 0]
    opt_new_bound = res_array[row_max, 1]

    if metric == "relative":
        opt_improvement = opt_standard_bound / opt_new_bound
        mean_improvement = mean_standard_bound / mean_new_bound
    elif metric == "absolute":
        opt_improvement = opt_standard_bound - opt_new_bound
        mean_improvement = mean_standard_bound - mean_new_bound
    else:
        raise NameError("Metric parameter {0} is infeasible".format(metric))

    res_dict = {
        "Name": "Value",
        "arrival_distribution": arrival.__class__.__name__
    }

    for j in range(number_servers):
        if isinstance(arrival, ExponentialArrival) and isinstance(
                service, ConstantRate):
            res_dict["lamb{0}".format(j + 1)] = param_array[row_max, j]
            res_dict["rate{0}".format(j + 1)] = param_array[row_max,
                                                            number_servers + j]

        elif isinstance(arrival, MMOO) and isinstance(service, ConstantRate):
            res_dict["mu{0}".format(j + 1)] = param_array[row_max, j]
            res_dict["lamb{0}".format(j + 1)] = param_array[row_max,
                                                            number_servers + j]
            res_dict["burst{0}".format(j + 1)] = param_array[
                row_max, 2 * number_servers + j]
            res_dict["rate{0}".format(j + 1)] = param_array[
                row_max, 3 * number_servers + j]

        else:
            raise NameError("Arrival parameter " + arrival.__class__.__name__ +
                            "or Service parameter" +
                            service.__class__.__name__ + " is infeasible")

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

    mean_ratio = lyapunov_mean / standard_mean

    time_ratio.update({number_servers: mean_ratio})

    # time_standard = [-2], time_lyapunov = [-1]
    res_dict = {
        "Name": "Value",
        "arrival_distribution": arrival.__class__.__name__,
        "number_servers": number_servers,
        "standard_mean": standard_mean,
        "Lyapunov_mean": lyapunov_mean,
        "mean_fraction": mean_ratio
    }

    return res_dict
