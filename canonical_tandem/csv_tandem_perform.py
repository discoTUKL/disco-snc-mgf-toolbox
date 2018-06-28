"""Compute delay bound for various T and write into csv file."""

import csv
from typing import List
from timeit import default_timer as timer

import pandas as pd

from canonical_tandem.tandem_sfa_perform import TandemSFA
from canonical_tandem.tandem_tfa_delay import TandemTFADelay
from library.perform_parameter import PerformParameter
from library.perform_param_list import PerformParamList
from nc_operations.nc_analysis import NCAnalysis
from nc_operations.perform_enum import PerformEnum
from nc_processes.arrival_distribution import ArrivalDistribution
from nc_processes.constant_rate_server import ConstantRate
from nc_processes.regulated_arrivals import (LeakyBucketMassOne,
                                             TokenBucketConstant)
from optimization.opt_method import OptMethod
from optimization.optimize import Optimize


def tandem_compare(arr_list: List[ArrivalDistribution],
                   arr_list2: List[ArrivalDistribution],
                   ser_list: List[ConstantRate], opt_method: OptMethod,
                   perform_param: PerformParameter,
                   nc_analysis: NCAnalysis) -> tuple:

    if nc_analysis == NCAnalysis.SFA:
        setting = TandemSFA(
            arr_list=arr_list, ser_list=ser_list, perform_param=perform_param)
        setting2 = TandemSFA(
            arr_list=arr_list2, ser_list=ser_list, perform_param=perform_param)
    elif nc_analysis == NCAnalysis.TFA and perform_param.perform_metric == PerformEnum.DELAY:
        setting = TandemTFADelay(
            arr_list=arr_list, ser_list=ser_list, prob_d=perform_param.value)
        setting2 = TandemTFADelay(
            arr_list=arr_list2, ser_list=ser_list, prob_d=perform_param.value)
    else:
        raise NameError(
            "{0} is an infeasible analysis type".format(nc_analysis))

    if opt_method == OptMethod.GRID_SEARCH:
        bound = Optimize(setting=setting).grid_search(
            bound_list=[(0.05, 15.0)], delta=0.05)
        bound2 = Optimize(setting=setting2).grid_search(
            bound_list=[(0.05, 15.0)], delta=0.05)
    else:
        raise ValueError(
            "Optimization parameter {0} is infeasible".format(opt_method))

    return bound, bound2


def csv_tandem_compare_servers(
        foi_arrival: ArrivalDistribution, cross_arrival: ArrivalDistribution,
        foi_arrival2: ArrivalDistribution, cross_arrival2: ArrivalDistribution,
        rate: float, max_servers: int, perform_param: PerformParameter,
        opt_method: OptMethod, nc_analysis: NCAnalysis) -> pd.DataFrame:
    """Write dataframe results into a csv file.

    Args:
        foi_arrival: flow of interest's arrival distribution
        foi_arrival2: competitor's flow of interest's arrival distribution
        cross_arrival: distribution of cross arrivals
        cross_arrival2: competitor's distribution of cross arrivals
        rate: service rate of servers
        max_servers: max number of servers in tandem
        perform_param: performance parameter values
        opt_method: optimization method
        nc_analysis: Network Calculus analysis type

    Returns:
        csv file

    """
    bounds = [0.0] * max_servers
    bounds2 = [0.0] * max_servers

    filename = "tandem_{0}".format(perform_param.to_name_value())

    arr_list: List[ArrivalDistribution] = [foi_arrival]
    arr_list2: List[ArrivalDistribution] = [foi_arrival2]
    ser_list: List[ConstantRate] = []

    for _i in range(max_servers):
        print("current_number_servers {0}".format(_i + 1))
        start = timer()
        arr_list.append(cross_arrival)
        arr_list2.append(cross_arrival2)
        ser_list.append(ConstantRate(rate=rate))

        bounds[_i], bounds2[_i] = tandem_compare(
            arr_list=arr_list,
            arr_list2=arr_list2,
            ser_list=ser_list,
            opt_method=opt_method,
            perform_param=perform_param,
            nc_analysis=nc_analysis)
        end = timer()
        print("duration: {0}".format(end - start))

    filename += "_max" + str(max_servers) + "servers_" + foi_arrival.to_value(
    ) + "_rate=" + str(rate)

    results_df = pd.DataFrame(
        {
            "bounds": bounds,
            "bounds2": bounds2
        },
        index=range(1, max_servers + 1))
    results_df = results_df[["bounds", "bounds2"]]

    results_df.to_csv(
        filename + '.csv', index=True, quoting=csv.QUOTE_NONNUMERIC)

    return results_df


def csv_tandem_compare_perform(
        foi_arrival: ArrivalDistribution, cross_arrival: ArrivalDistribution,
        foi_arrival2: ArrivalDistribution, cross_arrival2: ArrivalDistribution,
        rate: float, number_servers: int, perform_param_list: PerformParamList,
        opt_method: OptMethod, nc_analysis: NCAnalysis) -> pd.DataFrame:
    """Write dataframe results into a csv file.

    Args:
        foi_arrival: flow of interest's arrival distribution
        foi_arrival2: competitor's flow of interest's arrival distribution
        cross_arrival: distribution of cross arrivals
        cross_arrival2: competitor's distribution of cross arrivals
        rate: service rate of servers
        number_servers: number of servers in tandem
        perform_param_list: list of performance parameter values
        opt_method: optimization method
        nc_analysis: Network Calculus analysis type

    Returns:
        csv file

    """
    bounds = [0.0] * len(perform_param_list.values_list)
    bounds2 = [0.0] * len(perform_param_list.values_list)

    filename = "tandem_{0}".format(perform_param_list.to_name())

    arr_list: List[ArrivalDistribution] = [foi_arrival]
    arr_list2: List[ArrivalDistribution] = [foi_arrival2]
    ser_list: List[ConstantRate] = []

    for _i in range(number_servers):
        arr_list.append(cross_arrival)
        arr_list2.append(cross_arrival2)
        ser_list.append(ConstantRate(rate=rate))

    for _i in range(len(perform_param_list.values_list)):
        bounds[_i], bounds2[_i] = tandem_compare(
            arr_list=arr_list,
            arr_list2=arr_list2,
            ser_list=ser_list,
            opt_method=opt_method,
            perform_param=perform_param_list.get_parameter_at_i(_i),
            nc_analysis=nc_analysis)

    results_df = pd.DataFrame(
        {
            "bounds": bounds,
            "bounds2": bounds2
        },
        index=perform_param_list.values_list)
    results_df = results_df[["bounds", "bounds2"]]

    filename += "_" + str(number_servers) + "servers_" + foi_arrival.to_value(
    ) + "_rate=" + str(rate)

    results_df.to_csv(
        filename + '.csv', index=True, quoting=csv.QUOTE_NONNUMERIC)

    return results_df


if __name__ == '__main__':
    DELAY_LIST = PerformParamList(
        perform_metric=PerformEnum.DELAY,
        values_list=[10**(-1), 10**(-2), 10**(-4), 10**(-8), 10**(-12)])
    DELAY6 = PerformParameter(perform_metric=PerformEnum.DELAY, value=10**(-6))

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
        csv_tandem_compare_servers(
            foi_arrival=ARRIVAL_FOI,
            cross_arrival=ARRIVAL_CROSS,
            foi_arrival2=ARRIVAL_FOI2,
            cross_arrival2=ARRIVAL_CROSS2,
            rate=SERVICE_RATE,
            max_servers=10,
            perform_param=DELAY6,
            opt_method=OptMethod.GRID_SEARCH,
            nc_analysis=NCAnalysis.SFA))

    # print(
    #     csv_tandem_compare_perform(
    #         foi_arrival=ARRIVAL_FOI,
    #         cross_arrival=ARRIVAL_CROSS,
    #         foi_arrival2=ARRIVAL_FOI2,
    #         cross_arrival2=ARRIVAL_CROSS2,
    #         rate=SERVICE_RATE,
    #         number_servers=2,
    #         perform_param_list=DELAY_LIST,
    #         opt_method=OptMethod.GRID_SEARCH,
    #         nc_analysis=NCAnalysis.SFA))
