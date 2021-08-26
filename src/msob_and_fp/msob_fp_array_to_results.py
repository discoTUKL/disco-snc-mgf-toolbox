"""This file takes arrays and writes them into dictionaries"""

import csv
from math import inf

import numpy as np
from bound_evaluation.change_enum import ChangeEnum
from bound_evaluation.manipulate_data import remove_full_nan_rows
from bound_evaluation.monte_carlo_dist import MonteCarloDist
from nc_arrivals.arrival_enum import ArrivalEnum
from nc_operations.perform_enum import PerformEnum
from optimization.opt_method import OptMethod
from utils.exceptions import IllegalArgumentError
from utils.perform_parameter import PerformParameter


def msob_fp_array_to_results(title: str, arrival_enum: ArrivalEnum,
                             perform_param: PerformParameter,
                             opt_method: OptMethod, mc_dist: MonteCarloDist,
                             param_array: np.array, res_array: np.array,
                             number_flows: int, number_servers: int,
                             compare_metric: ChangeEnum) -> dict:
    """Writes the array values into a dictionary"""
    if res_array.shape[1] != 3:
        raise IllegalArgumentError(f"Array must have 3 columns,"
                                   f"not {res_array.shape[1]}")

    np.seterr(all='warn')

    res_array_no_full_nan = remove_full_nan_rows(full_array=res_array)
    valid_iterations = res_array_no_full_nan.shape[0]

    if compare_metric == ChangeEnum.RATIO_REF_NEW:
        change_vec_server_bound = np.divide(res_array[:, 0], res_array[:, 1])
        change_vec_pmoo_fp = np.divide(res_array[:, 0], res_array[:, 2])

    elif compare_metric == ChangeEnum.RATIO_NEW_REF:
        change_vec_server_bound = np.divide(res_array[:, 1], res_array[:, 0])
        change_vec_pmoo_fp = np.divide(res_array[:, 2], res_array[:, 0])

    elif compare_metric == ChangeEnum.RELATIVE_CHANGE:
        abs_vec_server_bound = np.subtract(res_array[:, 0], res_array[:, 1])
        change_vec_server_bound = np.divide(abs_vec_server_bound, res_array[:,
                                                                            0])

        abs_vec_pmoo_fp = np.subtract(res_array[:, 0], res_array[:, 2])
        change_vec_pmoo_fp = np.divide(abs_vec_pmoo_fp, res_array[:, 0])

    else:
        raise NotImplementedError(
            f"Metric={compare_metric.name} is not implemented")

    only_improved_server_bound = change_vec_server_bound[
        res_array[:, 0] > res_array[:, 1]]
    only_improved_pmoo_fp = change_vec_pmoo_fp[res_array[:, 0] > res_array[:,
                                                                           2]]

    row_max_msob = np.nanargmax(change_vec_server_bound)
    opt_msob = change_vec_server_bound[row_max_msob]
    mean_msob = np.nanmean(change_vec_server_bound)
    median_improved_server_bound = np.nanmedian(only_improved_server_bound)

    row_max_pmoo_fp = np.nanargmax(change_vec_pmoo_fp)
    opt_pmoo_fp = change_vec_pmoo_fp[row_max_pmoo_fp]
    mean_pmoo_fp = np.nanmean(change_vec_pmoo_fp)
    median_improved_pmoo_fp = np.nanmedian(only_improved_pmoo_fp)

    if (perform_param.perform_metric == PerformEnum.DELAY_PROB
            or perform_param.perform_metric == PerformEnum.BACKLOG_PROB):
        number_standard_bound_valid = np.nansum(
            res_array_no_full_nan[:, 0] < 1)
        number_server_bound_valid = np.nansum(res_array_no_full_nan[:, 1] < 1)
        number_pmoo_fp_valid = np.nansum(res_array_no_full_nan[:, 2] < 1)
    else:
        number_standard_bound_valid = np.nansum(
            res_array_no_full_nan[:, 0] < inf)
        number_server_bound_valid = np.nansum(
            res_array_no_full_nan[:, 1] < inf)
        number_pmoo_fp_valid = np.nansum(res_array_no_full_nan[:, 2] < inf)

    number_improved_server_bound = np.sum(
        res_array_no_full_nan[:, 0] > res_array_no_full_nan[:, 1])
    number_improved_pmoo_fp = np.sum(
        res_array_no_full_nan[:, 0] > res_array_no_full_nan[:, 2])

    best_approach = np.nanargmin(res_array_no_full_nan, axis=1)
    standard_best = np.count_nonzero(best_approach == 0)
    msob_best = np.count_nonzero(best_approach == 1)
    fp_best = np.count_nonzero(best_approach == 2)

    res_dict = {
        "Name": "Value",
        "topology": title,
        "arrival_distribution": arrival_enum.name
    }

    opt_dict = {
        "Name": "Value",
        "topology": title,
        "arrival_distribution": arrival_enum.name
    }

    for j in range(number_flows):
        if arrival_enum == ArrivalEnum.DM1:
            opt_dict[f"pmoo_fp_lamb{j + 1}"] = format(
                param_array[row_max_pmoo_fp, j], '.3f')
            opt_dict[f"server_bound_lamb{j + 1}"] = format(
                param_array[row_max_msob, j], '.3f')

        elif arrival_enum == ArrivalEnum.MD1:
            opt_dict[f"pmoo_fp_lamb{j + 1}"] = format(
                param_array[row_max_pmoo_fp, j], '.3f')
            opt_dict[f"ser_bound_lamb{j + 1}"] = format(
                param_array[row_max_msob, j], '.3f')

        elif arrival_enum == ArrivalEnum.MMOODisc:
            opt_dict[f"pmoo_fp_stay_on{j + 1}"] = format(
                param_array[row_max_pmoo_fp, j], '.3f')
            opt_dict[f"pmoo_fp_stay_off{j + 1}"] = format(
                param_array[row_max_pmoo_fp, number_flows + j], '.3f')
            opt_dict[f"pmoo_fp_burst{j + 1}"] = format(
                param_array[row_max_pmoo_fp, 2 * number_flows + j], '.3f')

            opt_dict[f"ser_bound_stay_on{j + 1}"] = format(
                param_array[row_max_msob, j], '.3f')
            opt_dict[f"ser_bound_stay_off{j + 1}"] = format(
                param_array[row_max_msob, number_flows + j], '.3f')
            opt_dict[f"ser_bound_burst{j + 1}"] = format(
                param_array[row_max_msob, 2 * number_flows + j], '.3f')

        elif arrival_enum == ArrivalEnum.MMOOFluid:
            opt_dict[f"pmoo_fp_mu{j + 1}"] = format(
                param_array[row_max_pmoo_fp, j], '.3f')
            opt_dict[f"pmoo_fp_lamb{j + 1}"] = format(
                param_array[row_max_pmoo_fp, number_flows + j], '.3f')
            opt_dict[f"pmoo_fp_burst{j + 1}"] = format(
                param_array[row_max_pmoo_fp, 2 * number_flows + j], '.3f')

            opt_dict[f"ser_bound_mu{j + 1}"] = format(
                param_array[row_max_msob, j], '.3f')
            opt_dict[f"ser_bound_lamb{j + 1}"] = format(
                param_array[row_max_msob, number_flows + j], '.3f')
            opt_dict[f"ser_bound_burst{j + 1}"] = format(
                param_array[row_max_msob, 2 * number_flows + j], '.3f')

        else:
            raise NotImplementedError(
                f"Arrival parameter={arrival_enum.name} is not implemented")

    for j in range(number_servers):
        opt_dict[f"pmoo_fp_rate{j + 1}"] = format(
            param_array[row_max_pmoo_fp,
                        arrival_enum.number_parameters() * number_flows + j],
            '.3f')
        opt_dict[f"server_bound_rate{j + 1}"] = format(
            param_array[row_max_msob,
                        arrival_enum.number_parameters() * number_flows + j],
            '.3f')

    opt_dict.update({
        "opt_pmoo_fp": format(opt_pmoo_fp, '.3f'),
        "opt_msob": format(opt_msob, '.3f'),
        "valid iterations": res_array.shape[0],
        "PerformParamValue": perform_param.value,
        "optimization": opt_method.name,
        "compare_metric": compare_metric.name,
        "MCDistribution": mc_dist.to_name(),
        "MCParam": mc_dist.param_to_string()
    })

    res_dict.update({
        "mean_pmoo_fp": mean_pmoo_fp,
        "mean_msob": mean_msob,
        "median_improved_pmoo_fp": median_improved_pmoo_fp,
        "median_improved_server_bound": median_improved_server_bound,
        "number standard bound is valid": number_standard_bound_valid,
        "number server bound is valid": number_server_bound_valid,
        "number PMOO_FP bound is valid": number_pmoo_fp_valid,
        "number server bound is improvement": number_improved_server_bound,
        "number PMOO_FP is improvement": number_improved_pmoo_fp,
        "valid iterations": valid_iterations,
        "number standard bound is best": standard_best,
        "number server bound is best": msob_best,
        "number PMOO_FP bound is best": fp_best,
    })

    filename = title
    filename += f"_optimal_{perform_param.to_name()}_{arrival_enum.name}_" \
                f"MC{mc_dist.to_name()}_{opt_method.name}_" \
                f"{compare_metric.name}"

    with open(filename + ".csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in opt_dict.items():
            writer.writerow([key, value])

    return res_dict


def time_array_to_results(title: str, time_array: np.array) -> dict:
    """Writes the array values into a dictionary"""
    mean_standard = np.nanmean(time_array[:, 0])
    mean_server_bound = np.nanmean(time_array[:, 1])
    mean_pmoo_fp = np.nanmean(time_array[:, 2])

    median_standard = np.nanmedian(time_array[:, 0])
    median_server_bound = np.nanmedian(time_array[:, 1])
    median_pmoo_fp = np.nanmedian(time_array[:, 2])

    ratio_server_standard = np.divide(time_array[:, 0], time_array[:, 1])
    ratio_fp_standard = np.divide(time_array[:, 0], time_array[:, 2])
    mean_ratio_server_standard = np.nanmean(ratio_server_standard)
    mean_ratio_fp_standard = np.nanmean(ratio_fp_standard)
    median_ratio_server_standard = np.nanmedian(ratio_server_standard)
    median_ratio_fp_standard = np.nanmedian(ratio_fp_standard)

    res_dict = {
        "Name": "Value",
        "topology": title,
        "mean_standard": format(mean_standard, '.4f'),
        "mean_server_bound": format(mean_server_bound, '.4f'),
        "mean_pmoo_fp": format(mean_pmoo_fp, '.4f'),
        "mean_ratio_server_standard": format(mean_ratio_server_standard,
                                             '.4f'),
        "mean_ratio_fp_standard": format(mean_ratio_fp_standard, '.4f'),
        "median_standard": format(median_standard, '.4f'),
        "median_server_bound": format(median_server_bound, '.4f'),
        "median_pmoo_fp": format(median_pmoo_fp, '.4f'),
        "median_ratio_server_standard": format(median_ratio_server_standard,
                                               '.4f'),
        "median_ratio_fp_standard": format(median_ratio_fp_standard, '.4f')
    }

    return res_dict
