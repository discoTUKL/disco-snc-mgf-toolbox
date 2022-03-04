"""This file takes arrays and writes them into dictionaries"""

from warnings import warn

import numpy as np
from bound_evaluation.change_enum import ChangeEnum
from bound_evaluation.manipulate_data import remove_full_nan_rows
from nc_arrivals.arrival_enum import ArrivalEnum
from utils.exceptions import IllegalArgumentError


def two_col_array_to_results(
        arrival_enum: ArrivalEnum,
        param_array: np.array,
        res_array: np.array,
        number_servers: int,
        compare_metric: ChangeEnum = ChangeEnum.RATIO_REF_NEW) -> dict:
    """Writes the array values into a dictionary"""
    if res_array.shape[1] != 2:
        raise IllegalArgumentError(
            f"Array must have 2 columns, not {res_array.shape[1]}")

    np.seterr(all='warn')

    res_array_no_full_nan = remove_full_nan_rows(full_array=res_array)
    valid_iterations = res_array_no_full_nan.shape[0]

    if compare_metric == ChangeEnum.RATIO_REF_NEW:
        improvement_vec = np.divide(res_array[:, 0], res_array[:, 1])
    elif compare_metric == ChangeEnum.DIFF_REF_NEW:
        improvement_vec = np.subtract(res_array[:, 0], res_array[:, 1])
    else:
        raise NotImplementedError(
            f"Metric={compare_metric.name} is not implemented")

    row_max = np.nanargmax(improvement_vec)

    opt_standard_bound = res_array[row_max, 0]
    opt_h_mit_bound = res_array[row_max, 1]
    opt_improvement = improvement_vec[row_max]

    mean_improvement = np.nanmean(improvement_vec)
    median_improvement = np.nanmedian(improvement_vec)

    number_improved = np.sum(res_array[:, 0] > res_array[:, 1])

    res_dict = {"Name": "Value", "arrival_distribution": arrival_enum.name}

    for j in range(number_servers):
        if arrival_enum == ArrivalEnum.DM1:
            res_dict[f"lamb{j + 1}"] = format(param_array[row_max, j], '.3f')
            res_dict[f"rate{j + 1}"] = format(
                param_array[row_max, number_servers + j], '.3f')

        elif arrival_enum == ArrivalEnum.DGamma1:
            res_dict[f"alpha_shape{j + 1}"] = format(param_array[row_max, j],
                                                     '.3f')
            res_dict[f"beta_rate{j + 1}"] = format(
                param_array[row_max, number_servers + j], '.3f')
            res_dict[f"rate{j + 1}"] = format(
                param_array[row_max, 2 * number_servers + j], '.3f')

        elif arrival_enum == ArrivalEnum.DWeibull1:
            res_dict[f"lamb{j + 1}"] = format(param_array[row_max, j], '.3f')
            res_dict[f"rate{j + 1}"] = format(
                param_array[row_max, number_servers + j], '.3f')

        elif arrival_enum == ArrivalEnum.MD1:
            res_dict[f"lamb{j + 1}"] = format(param_array[row_max, j], '.3f')
            res_dict[f"rate{j + 1}"] = format(
                param_array[row_max, number_servers + j], '.3f')
            res_dict[f"packet_size{j + 1}"] = format(
                param_array[row_max, number_servers + j], '.3f')

        elif arrival_enum == ArrivalEnum.MMOODisc:
            res_dict[f"stay_on{j + 1}"] = format(param_array[row_max, j],
                                                 '.3f')
            res_dict[f"stay_off{j + 1}"] = format(
                param_array[row_max, number_servers + j], '.3f')
            res_dict[f"peak_rate{j + 1}"] = format(
                param_array[row_max, 2 * number_servers + j], '.3f')
            res_dict[f"rate{j + 1}"] = format(
                param_array[row_max, 3 * number_servers + j], '.3f')

        elif arrival_enum == ArrivalEnum.MMOOFluid:
            res_dict[f"mu{j + 1}"] = format(param_array[row_max, j], '.3f')
            res_dict[f"lamb{j + 1}"] = format(
                param_array[row_max, number_servers + j], '.3f')
            res_dict[f"peak_rate{j + 1}"] = format(
                param_array[row_max, 2 * number_servers + j], '.3f')
            res_dict[f"rate{j + 1}"] = format(
                param_array[row_max, 3 * number_servers + j], '.3f')

        elif arrival_enum == ArrivalEnum.EBB:
            res_dict[f"M{j + 1}"] = format(param_array[row_max, j], '.3f')
            res_dict[f"b{j + 1}"] = format(
                param_array[row_max, number_servers + j], '.3f')
            res_dict[f"rho{j + 1}"] = format(
                param_array[row_max, 2 * number_servers + j], '.3f')
            res_dict[f"rate{j + 1}"] = format(
                param_array[row_max, 3 * number_servers + j], '.3f')

        else:
            raise NotImplementedError(
                f"Arrival parameter={arrival_enum.name} is not implemented")

    res_dict.update({
        "opt standard standard_bound": opt_standard_bound,
        "opt h-mitigator standard_bound": opt_h_mit_bound,
        "optimum improvement": format(opt_improvement, '.3f'),
        "mean improvement": mean_improvement,
        "median improvement": median_improvement,
        "number improved": number_improved,
        "valid iterations": valid_iterations,
        "share improved": number_improved / valid_iterations
    })

    return res_dict


def three_col_array_to_results(
        arrival_enum: ArrivalEnum,
        res_array: np.array,
        valid_iterations: int,
        compare_metric: ChangeEnum = ChangeEnum.RATIO_REF_NEW) -> dict:
    """Writes the array values into a dictionary"""
    if res_array.shape[1] != 3:
        raise NameError(f"Array must have 3 columns, not {res_array.shape[1]}")

    iterations = int(res_array.shape[0])

    if compare_metric == ChangeEnum.RATIO_REF_NEW:
        improvement_vec_1 = np.divide(res_array[:, 0], res_array[:, 1])
        improvement_vec_2 = np.divide(res_array[:, 0], res_array[:, 2])
        improvement_vec_news = np.divide(res_array[:, 1], res_array[:, 2])
    elif compare_metric == ChangeEnum.DIFF_REF_NEW:
        improvement_vec_1 = np.subtract(res_array[:, 0], res_array[:, 1])
        improvement_vec_2 = np.subtract(res_array[:, 0], res_array[:, 2])
        improvement_vec_news = np.subtract(res_array[:, 1], res_array[:, 2])
    else:
        raise NotImplementedError(
            f"Metric={compare_metric.name} is not implemented")

    row_exp_max = np.nanargmax(improvement_vec_news)

    opt_standard_bound = res_array[row_exp_max, 0]
    opt_power_bound = res_array[row_exp_max, 1]
    opt_exp_bound = res_array[row_exp_max, 2]

    # opt_power_improvement = improvement_vec_1[row_exp_max]
    # opt_exp_improvement = improvement_vec_2[row_exp_max]
    opt_new_improvement = improvement_vec_news[row_exp_max]

    mean_power_improvement = np.nanmean(improvement_vec_1)
    mean_exp_improvement = np.nanmean(improvement_vec_2)
    mean_new_improvement = np.nanmean(improvement_vec_news)

    median_power_improvement = np.nanmedian(improvement_vec_1)
    median_exp_improvement = np.nanmedian(improvement_vec_2)
    median_new_improvement = np.nanmedian(improvement_vec_news)

    # number_improved = np.sum(np.greater(res_array[:, 1], res_array[:, 2]))
    number_improved = np.sum(res_array[:, 1] > res_array[:, 2])

    if valid_iterations < iterations * 0.2:
        warn(f"way too many nan's: "
             f"{iterations - valid_iterations} nan out of {iterations}!")

        if valid_iterations < 100:
            raise ValueError("result is useless")

    res_dict = {"Name": "Value", "arrival_distribution": arrival_enum.name}

    res_dict.update({
        "opt_standard_bound": opt_standard_bound,
        "opt_power_bound": opt_power_bound,
        "opt_exp_bound": opt_exp_bound,
        # "opt_power_improvement": opt_power_improvement,
        # "opt_exp_improvement": opt_exp_improvement,
        "opt_new_improvement": format(opt_new_improvement, '.3f'),
        "mean_power_improvement": mean_power_improvement,
        "mean_exp_improvement": mean_exp_improvement,
        "mean_new_improvement": mean_new_improvement,
        "median_power_improvement": median_power_improvement,
        "median_exp_improvement": median_exp_improvement,
        "median_new_improvement": median_new_improvement,
        "number improved": number_improved,
        "valid iterations": valid_iterations,
        "share improved": number_improved / valid_iterations
    })

    return res_dict


def time_array_to_results(arrival_enum: ArrivalEnum, time_array,
                          number_servers: int, time_ratio: dict) -> dict:
    """Writes the array values into a dictionary"""

    standard_mean = np.nanmean(time_array[:, 0])
    power_mean = np.nanmean(time_array[:, 1])

    ratio = np.divide(time_array[:, 1], time_array[:, 0])
    mean_ratio = np.nanmean(ratio)

    time_ratio.update({number_servers: mean_ratio})

    res_dict = {
        "Name": "Value",
        "arrival_distribution": arrival_enum.name,
        "number_servers": number_servers,
        "standard_mean": format(standard_mean, '.4f'),
        "power_mean": format(power_mean, '.4f'),
        "mean_fraction": format(mean_ratio, '.3f')
    }

    return res_dict
