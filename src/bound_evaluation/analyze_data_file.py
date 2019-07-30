"""Remove rows that contain nan-values"""

import csv

import numpy as np

from bound_evaluation.change_enum import ChangeEnum
from bound_evaluation.manipulate_data import remove_nan_rows
from nc_arrivals.arrival_enum import ArrivalEnum
from nc_operations.perform_enum import PerformEnum
from utils.perform_parameter import PerformParameter


def remove_all_nan_file(filename_csv: str) -> np.array:
    res_array = np.genfromtxt(filename_csv + ".csv", delimiter=",")

    res_array_no_nan = remove_nan_rows(full_array=res_array)

    np.savetxt(filename_csv + "_no_nan.csv", X=res_array_no_nan, delimiter=',')

    return res_array_no_nan


def remove_all_inf_file(filename_csv: str) -> np.array:
    res_array = np.genfromtxt(filename_csv + ".csv", delimiter=",")

    res_array_no_nan = res_array[~np.isinf(res_array).any(axis=1)]

    np.savetxt(filename_csv + "_no_inf.csv", X=res_array_no_nan, delimiter=',')

    return res_array_no_nan


def remove_all_inf_ignore_server_file(filename_csv: str) -> np.array:
    res_array = np.genfromtxt(filename_csv + ".csv", delimiter=",")

    res_array = np.delete(res_array, obj=3, axis=1)

    res_array_no_nan = res_array[~np.isinf(res_array).any(axis=1)]

    np.savetxt(filename_csv + "_no_inf_ignore_server.csv",
               X=res_array_no_nan,
               delimiter=',')

    return res_array_no_nan


def remove_standard_inf_file(filename_csv: str) -> np.array:
    res_array = np.genfromtxt(filename_csv + ".csv", delimiter=",")

    res_array_no_nan = res_array[res_array[:, 0] < np.inf]

    np.savetxt(filename_csv + "_no_standard_inf.csv",
               X=res_array_no_nan,
               delimiter=',')

    return res_array_no_nan


def results_ignore_server(arrival_enum: ArrivalEnum,
                          perform_param: PerformParameter,
                          metric: ChangeEnum) -> dict:
    """Writes the array values into a dictionary"""
    filename_csv = f"res_array_{perform_param.to_name()}_{arrival_enum.name}_" \
        f"bounds_{metric.name}_no_nan_no_inf_ignore_server"

    res_array = np.genfromtxt(filename_csv + ".csv", delimiter=",")

    if res_array.shape[1] != 3:
        raise NameError(f"Array must have 3 columns, not {res_array.shape[1]}")

    remaining_iterations = int(res_array.shape[0])

    np.seterr(all='warn')

    if metric == ChangeEnum.RATIO_REF_NEW:
        change_vec_neg_dep_avoid_hoelder = np.divide(res_array[:, 0],
                                                     res_array[:, 1])
        change_vec_pmoo_fp = np.divide(res_array[:, 0], res_array[:, 2])
    elif metric == ChangeEnum.RATIO_NEW_REF:
        change_vec_neg_dep_avoid_hoelder = np.divide(res_array[:, 1],
                                                     res_array[:, 0])
        change_vec_pmoo_fp = np.divide(res_array[:, 2], res_array[:, 0])
    elif metric == ChangeEnum.RELATIVE_CHANGE:
        abs_vec_neg_dep_avoid_hoelder = np.subtract(res_array[:, 0],
                                                    res_array[:, 1])
        change_vec_neg_dep_avoid_hoelder = np.divide(
            abs_vec_neg_dep_avoid_hoelder, res_array[:, 0])

        abs_vec_pmoo_fp = np.subtract(res_array[:, 0], res_array[:, 2])
        change_vec_pmoo_fp = np.divide(abs_vec_pmoo_fp, res_array[:, 0])
    else:
        raise NotImplementedError(f"Metric={metric.name} is not implemented")

    opt_neg_dep_avoid_hoelder = np.nanmax(change_vec_neg_dep_avoid_hoelder)
    opt_pmoo_fp = np.nanmax(change_vec_pmoo_fp)

    mean_neg_dep_avoid_hoelder = np.nanmean(change_vec_neg_dep_avoid_hoelder)
    mean_pmoo_fp = np.nanmean(change_vec_pmoo_fp)

    number_improved_neg_avoid_hoelder = np.sum(
        res_array[:, 0] > res_array[:, 1])
    number_improved_pmoo_fp = np.sum(res_array[:, 0] > res_array[:, 2])

    res_dict = {"Name": "Value", "arrival_distribution": arrival_enum.name}

    res_dict.update({
        "opt_neg_dep_avoid_hoelder":
        opt_neg_dep_avoid_hoelder,
        "opt_pmoo_fp":
        opt_pmoo_fp,
        "mean_neg_dep_avoid_hoelder":
        mean_neg_dep_avoid_hoelder,
        "mean_pmoo_fp":
        mean_pmoo_fp,
        "number avoiding Hoelder with negative dep. is improvement":
        number_improved_neg_avoid_hoelder,
        "number PMOO_FP is improvement":
        number_improved_pmoo_fp,
        "remaining_iterations":
        remaining_iterations,
        "metric":
        metric.name
    })

    with open(f"res_array_analysis_ignore_server_{metric.name}.csv",
              'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in res_dict.items():
            writer.writerow([key, value])

    return res_dict


if __name__ == '__main__':
    DELAY_PROB10 = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                    value=10)
    DM1 = ArrivalEnum.DM1
    METRIC = ChangeEnum.RATIO_REF_NEW
    # remove_all_nan_file(
    #     filename_csv=f"res_array_{DELAY_PROB10.to_name()}_{DM1.name}_"
    #     f"bounds_{METRIC.name}")

    # remove_all_inf_ignore_server_file(
    #     filename_csv=f"res_array_{DELAY_PROB10.to_name()}_{DM1.name}_"
    #     f"bounds_{METRIC.name}_no_nan")

    # remove_standard_inf_file(
    #     filename_csv=f"res_array_{DELAY_PROB10.to_name()}_{DM1.name}_"
    #     f"bounds_{METRIC.name}_no_nan")

    # remove_all_inf_file(
    #     filename_csv=f"res_array_{DELAY_PROB10.to_name()}_{DM1.name}_"
    #     f"bounds_{METRIC.name}_no_nan")

    results_ignore_server(arrival_enum=DM1,
                          perform_param=DELAY_PROB10,
                          metric=METRIC)
