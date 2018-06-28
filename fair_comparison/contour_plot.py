"""See how much multiplexing is necessary to beat the DNC bound"""

import csv
from typing import List

import pandas as pd

from library.perform_parameter import PerformParameter
from nc_operations.dnc_performance_bounds import FIFODelay
from nc_operations.perform_metric import PerformMetric
from nc_processes.regulated_arrivals import (LeakyBucketMassOne,
                                             TokenBucketConstant)
from nc_processes.constant_rate_server import ConstantRate
from optimization.optimize import Optimize
from single_server.single_server_perform import SingleServerPerform


def single_hop_contour(sigma_single: float,
                       rho_single: float,
                       utilization: float,
                       perform_param: PerformParameter,
                       pure_snc=False) -> int:
    print_x = False

    bound_list = [(0.05, 15.0)]
    delta = 0.05

    aggregation = 1

    # util = n * rho / service => service = n * rho / util

    constant_rate_server = ConstantRate(aggregation * rho_single / utilization)

    tb_const = TokenBucketConstant(
        sigma_single=sigma_single, rho_single=rho_single, n=aggregation)

    if pure_snc:
        competitor = Optimize(
            setting=SingleServerPerform(
                arr=tb_const,
                const_rate=constant_rate_server,
                perform_param=perform_param),
            print_x=print_x).grid_search(
                bound_list=bound_list, delta=delta)

    else:
        competitor = FIFODelay(
            token_bucket_constant=tb_const, constant_rate=constant_rate_server)

    leaky_mass_1_opt = Optimize(
        setting=SingleServerPerform(
            arr=LeakyBucketMassOne(
                sigma_single=sigma_single,
                rho_single=rho_single,
                n=aggregation),
            const_rate=constant_rate_server,
            perform_param=perform_param),
        print_x=print_x).grid_search(
            bound_list=bound_list, delta=delta)

    while competitor < leaky_mass_1_opt:
        aggregation += 1

        constant_rate_server = ConstantRate(
            aggregation * rho_single / utilization)

        tb_const = TokenBucketConstant(
            sigma_single=sigma_single, rho_single=rho_single, n=aggregation)

        if pure_snc:
            competitor = Optimize(
                setting=SingleServerPerform(
                    arr=tb_const,
                    const_rate=constant_rate_server,
                    perform_param=perform_param),
                print_x=print_x).grid_search(
                    bound_list=bound_list, delta=delta)

        else:
            competitor = FIFODelay(
                token_bucket_constant=tb_const,
                constant_rate=constant_rate_server)

        leaky_mass_1_opt = Optimize(
            setting=SingleServerPerform(
                arr=LeakyBucketMassOne(
                    sigma_single=sigma_single,
                    rho_single=rho_single,
                    n=aggregation),
                const_rate=constant_rate_server,
                perform_param=perform_param),
            print_x=print_x).grid_search(
                bound_list=bound_list, delta=delta)

    # print("(dnc_fifo_single, const_opt, leaky_mass_1_opt)")
    # print(dnc_fifo_single, const_opt, leaky_mass_1_opt)

    return aggregation


def csv_contour(rho_single: float,
                sigma_list: List[float],
                utilization: float,
                perform_param: PerformParameter,
                pure_snc=False) -> pd.DataFrame:
    agg_list = [0] * len(sigma_list)

    for i, sigma in enumerate(sigma_list):
        agg_list[i] = single_hop_contour(
            sigma_single=sigma,
            rho_single=rho_single,
            utilization=utilization,
            perform_param=perform_param,
            pure_snc=pure_snc)

    results_df = pd.DataFrame(
        {
            "aggregation": agg_list,
        }, index=sigma_list)

    if pure_snc:
        filename = "contour_{0}_rho_{1}_utilization_{2}_pure".format(
            perform_param.to_name_value(), str(rho_single),
            str("%.2f" % utilization))
    else:
        filename = "contour_{0}_rho_{1}_utilization_{2}".format(
            perform_param.to_name_value(), str(rho_single),
            str("%.2f" % utilization))

    results_df.to_csv(
        filename + '.csv', index=True, quoting=csv.QUOTE_NONNUMERIC)

    return results_df


if __name__ == '__main__':
    DELAY6 = PerformParameter(
        perform_metric=PerformMetric.DELAY, value=10**(-6))

    NUMBER_AGGREGATIONS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50]

    RHO_SINGLE = 0.1
    UTILIZATION = RHO_SINGLE / 0.12

    SIGMA_VALUES_1 = [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 300.0]

    # for SIGMA in SIGMA_VALUES_1:
    # print(
    #     single_hop_contour(
    #         sigma_single=SIGMA,
    #         rho_single=RHO_SINGLE,
    #         utilization=UTILIZATION,
    #         perform_param=DELAY6))

    print(
        csv_contour(
            rho_single=RHO_SINGLE,
            sigma_list=SIGMA_VALUES_1,
            utilization=UTILIZATION,
            perform_param=DELAY6,
            pure_snc=False))

    print(
        csv_contour(
            rho_single=RHO_SINGLE,
            sigma_list=SIGMA_VALUES_1,
            utilization=UTILIZATION,
            perform_param=DELAY6,
            pure_snc=True))
