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
from nc_processes.constant_rate_server import ConstantRate
from nc_processes.regulated_arrivals import (LeakyBucketMassOne,
                                             TokenBucketConstant)
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize


def tandem_df(arr_list: List[ArrivalDistribution],
              arr_list2: List[ArrivalDistribution],
              ser_list: List[ConstantRate], opt_method: OptMethod,
              perform_param_list: PerformParamList,
              nc_analysis: NCAnalysis) -> pd.DataFrame:
    bounds = [0.0] * len(perform_param_list.values_list)
    bounds2 = [0.0] * len(perform_param_list.values_list)

    for _i in range(len(perform_param_list.values_list)):
        perform_param = perform_param_list.get_parameter_at_i(_i)

        if nc_analysis == NCAnalysis.SFA:
            setting = TandemSFA(
                arr_list=arr_list,
                ser_list=ser_list,
                perform_param=perform_param)
            setting2 = TandemSFA(
                arr_list=arr_list2,
                ser_list=ser_list,
                perform_param=perform_param)
        elif nc_analysis == NCAnalysis.TFA and perform_param_list.perform_metric == PerformMetric.DELAY:
            setting = TandemTFADelay(
                arr_list=arr_list,
                ser_list=ser_list,
                prob_d=perform_param_list.values_list[_i])
            setting2 = TandemTFADelay(
                arr_list=arr_list2,
                ser_list=ser_list,
                prob_d=perform_param_list.values_list[_i])
        else:
            raise NameError(
                "{0} is an infeasible analysis type".format(nc_analysis))

        if opt_method == OptMethod.GRID_SEARCH:
            bounds[_i] = Optimize(setting=setting).grid_search(
                bound_list=[(0.05, 15.0)], delta=0.05)
            bounds2[_i] = Optimize(setting=setting2).grid_search(
                bound_list=[(0.05, 15.0)], delta=0.05)
        else:
            raise ValueError(
                "Optimization parameter {0} is infeasible".format(opt_method))

    results_df = pd.DataFrame(
        {
            "bounds": bounds,
            "bounds2": bounds2
        },
        index=perform_param_list.values_list)
    results_df = results_df[["bounds", "bounds2"]]

    return results_df


def csv_tandem_perform(
        foi_arrival: ArrivalDistribution, cross_arrival: ArrivalDistribution,
        foi_arrival2: ArrivalDistribution, cross_arrival2: ArrivalDistribution,
        const_rate: ConstantRate, number_servers: int,
        perform_param_list: PerformParamList, opt_method: OptMethod,
        nc_analysis: NCAnalysis) -> pd.DataFrame:
    """Write dataframe results into a csv file.

    Args:
        foi_arrival: flow of interest's arrival distribution
        foi_arrival2: competitor's flow of interest's arrival distribution
        cross_arrival: distribution of cross arrivals
        cross_arrival2: competitor's distribution of cross arrivals
        const_rate: service of remaining servers
        number_servers: number of servers in fat tree
        perform_param_list: list of performance parameter values
        opt_method: optimization method
        nc_analysis: Network Calculus analysis type

    Returns:
        csv file

    """
    filename = "tandem_{0}".format(perform_param_list.perform_metric.name)

    arr_list: List[ArrivalDistribution] = [foi_arrival]
    arr_list2: List[ArrivalDistribution] = [foi_arrival2]
    ser_list: List[ConstantRate] = []

    for _i in range(number_servers):
        arr_list.append(cross_arrival)
        arr_list2.append(cross_arrival2)
        ser_list.append(const_rate)

    data_frame = tandem_df(
        arr_list=arr_list,
        arr_list2=arr_list2,
        ser_list=ser_list,
        opt_method=opt_method,
        perform_param_list=perform_param_list,
        nc_analysis=nc_analysis)

    filename += "_" + str(number_servers) + "servers_" + foi_arrival.to_string(
    ) + "_" + cross_arrival.to_string() + "_" + const_rate.to_string()

    data_frame.to_csv(
        filename + '.csv', index=True, quoting=csv.QUOTE_NONNUMERIC)

    return data_frame


if __name__ == '__main__':
    DELAY_LIST = PerformParamList(
        perform_metric=PerformMetric.DELAY,
        values_list=[10**(-1), 10**(-2), 10**(-4), 10**(-8), 10**(-12)])

    NUMBER_AGGREGATIONS = 10
    RHO_SINGLE = 3.0
    SERVICE_RATE = 70.0
    SIGMA_SINGLE = 10.0

    ARRIVAL_FOI = LeakyBucketMassOne(
        sigma_single=SIGMA_SINGLE,
        rho_single=RHO_SINGLE,
        n=NUMBER_AGGREGATIONS)
    ARRIVAL_CROSS = LeakyBucketMassOne(
        sigma_single=SIGMA_SINGLE,
        rho_single=RHO_SINGLE,
        n=NUMBER_AGGREGATIONS)
    ARRIVAL_FOI2 = TokenBucketConstant(
        sigma_single=SIGMA_SINGLE,
        rho_single=RHO_SINGLE,
        n=NUMBER_AGGREGATIONS)
    ARRIVAL_CROSS2 = TokenBucketConstant(
        sigma_single=SIGMA_SINGLE,
        rho_single=RHO_SINGLE,
        n=NUMBER_AGGREGATIONS)
    print(
        csv_tandem_perform(
            foi_arrival=ARRIVAL_FOI,
            cross_arrival=ARRIVAL_CROSS,
            foi_arrival2=ARRIVAL_FOI2,
            cross_arrival2=ARRIVAL_CROSS2,
            const_rate=ConstantRate(rate=SERVICE_RATE),
            number_servers=2,
            perform_param_list=DELAY_LIST,
            opt_method=OptMethod.GRID_SEARCH,
            nc_analysis=NCAnalysis.SFA))
