"""Compute delay bound for various T and write into csv file."""

import csv
from typing import List

import pandas as pd

from canonical_tandem.tandem_sfa_perform import TandemSFA
from canonical_tandem.tandem_tfa_delay import TandemTFADelay
from library.perform_param_list import PerformParamList
from nc_operations.nc_analysis import NCAnalysis
from nc_operations.perform_metric import PerformMetric
from nc_processes.arrival_distribution import ArrivalDistribution
from nc_processes.regulated_arrivals import RegulatedArrivals
from nc_processes.service_distribution import ConstantRate, ServiceDistribution
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize


def tandem_df(arr_list: List[ArrivalDistribution],
              ser_list: List[ServiceDistribution], opt_method: OptMethod,
              perform_param_list: PerformParamList,
              nc_analysis: NCAnalysis) -> pd.DataFrame:
    bounds = [0.0] * len(perform_param_list.values_list)

    for _i in range(len(perform_param_list.values_list)):
        perform_param = perform_param_list.get_parameter_at_i(_i)

        if nc_analysis == NCAnalysis.SFA:
            setting = TandemSFA(
                arr_list=arr_list,
                ser_list=ser_list,
                perform_param=perform_param)
        elif nc_analysis == NCAnalysis.TFA and perform_param_list.perform_metric == PerformMetric.DELAY:
            setting = TandemTFADelay(
                arr_list=arr_list,
                ser_list=ser_list,
                prob_d=perform_param_list.values_list[_i])
        else:
            raise NameError("{0} is an infeasible analysis type".format(
                nc_analysis))

        if opt_method == OptMethod.GRID_SEARCH:
            bounds[_i] = Optimize(setting=setting).grid_search(
                bound_list=[(0.05, 15.0)], delta=0.05)
        else:
            raise ValueError(
                "Optimization parameter {0} is infeasible".format(opt_method))

    results_df = pd.DataFrame(
        {
            "bounds": bounds
        }, index=perform_param_list.values_list)
    results_df = results_df[["bounds"]]

    return results_df


# def csv_tandem_perform(reg_arrival: RegulatedArrivals,
#                        list_of_parameters: List[DistribParam],
#                        number_servers: int,
#                        perform_param_list: PerformParamList,
#                        opt_method: OptMethod,
#                        nc_analysis: NCAnalysis) -> pd.DataFrame:
#     """Write dataframe results into a csv file.
#
#     Args:
#         arrival: String that represents the arrival process
#         list_of_parameters: dictionaries with actual values
#         perform_param_list: list of performance parameter values
#         opt_method: optimization method
#         nc_analysis: Network Calculus analysis
#
#     Returns:
#         csv file
#
#     """
#     filename = "tandem_{0}".format(perform_param_list.perform_metric.name)
#
#     sigma1 = list_of_parameters[0].sigma
#     sigma2 = list_of_parameters[1].sigma
#     rho1 = list_of_parameters[0].rho
#     rho2 = list_of_parameters[1].rho
#     homogeneous_rate = list_of_parameters[2].rate
#
#     data_frame = tandem_df(
#         arr_list=[
#             reg_arrival(mu=mu1, lamb=lamb1, burst=burst1),
#             MMOO(mu=mu2, lamb=lamb2, burst=burst2)
#         ],
#         ser_list=[ConstantRate(rate=rate1),
#                   ConstantRate(rate=rate2)],
#         opt_method=opt_method,
#         perform_param_list=perform_param_list,
#         nc_analysis=nc_analysis)
#
#     for item, value in enumerate(list_of_parameters):
#         filename += "_" + value.get_mmoo_string(item)
#
#     data_frame.to_csv(
#         filename + '.csv', index=True, quoting=csv.QUOTE_NONNUMERIC)
#
#     return data_frame
