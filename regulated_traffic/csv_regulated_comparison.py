"""Compare Delay for different bounds for regulated Arrivals"""

import csv
from typing import List

import pandas as pd

from dnc.dnc_fifo_delay import DNCFIFODelay
from library.perform_parameter import PerformParameter
from nc_operations.perform_metric import PerformMetric
from nc_processes.arrival_distribution import (
    LeakyBucketMassOne, LeakyBucketMassTwo, LeakyBucketMassTwoExact,
    TokenBucketConstant)
from nc_processes.service import ConstantRate
from optimization.optimize import Optimize
from single_server.single_server_perform import SingleServerPerform


def regulated_comparison(aggregation: int, sigma_single: float,
                         rho_single: float, service_rate: float,
                         perform_param: PerformParameter) -> tuple():
    constant_rate_server = ConstantRate(service_rate)
    tb_const = TokenBucketConstant(
        sigma_const=sigma_single, rho_const=rho_single, n=aggregation)

    bound_list = [(0.05, 15.0)]
    delta = 0.05

    dnc_fifo_single = DNCFIFODelay(
        token_bucket_constant=tb_const,
        constant_rate=constant_rate_server).bound()

    const_single = SingleServerPerform(
        arr=tb_const, ser=constant_rate_server, perform_param=perform_param)
    const_opt = Optimize(
        setting=const_single, print_x=False).grid_search_old(
            bound_list=bound_list, delta=delta)

    leaky_mass_1 = SingleServerPerform(
        arr=LeakyBucketMassOne(
            sigma_single=sigma_single, rho_single=rho_single, n=aggregation),
        ser=constant_rate_server,
        perform_param=perform_param)
    leaky_mass_1 = Optimize(
        setting=leaky_mass_1, print_x=False).grid_search_old(
            bound_list=bound_list, delta=delta)

    leaky_mass_2 = SingleServerPerform(
        arr=LeakyBucketMassTwo(
            sigma_single=sigma_single, rho_single=rho_single, n=aggregation),
        ser=constant_rate_server,
        perform_param=perform_param)
    leaky_mass_2_opt = Optimize(
        setting=leaky_mass_2, print_x=False).grid_search_old(
            bound_list=bound_list, delta=delta)

    exact_mass_2 = SingleServerPerform(
        arr=LeakyBucketMassTwoExact(
            sigma_single=sigma_single, rho_single=rho_single, n=aggregation),
        ser=constant_rate_server,
        perform_param=perform_param)
    exact_mass_2_opt = Optimize(
        setting=exact_mass_2, print_x=False).grid_search_old(
            bound_list=bound_list, delta=delta)

    return dnc_fifo_single, const_opt, leaky_mass_1, leaky_mass_2_opt,\
           exact_mass_2_opt


def compare_aggregation(aggregations: List[int], sigma_single: float,
                        rho_single: float, service_rate: float,
                        perform_param: PerformParameter) -> pd.DataFrame:
    dnc_fifo_single = [0.0] * len(aggregations)
    const_opt = [0.0] * len(aggregations)
    leaky_mass_1 = [0.0] * len(aggregations)
    leaky_mass_2_opt = [0.0] * len(aggregations)
    exact_mass_2_opt = [0.0] * len(aggregations)

    for i, agg in enumerate(aggregations):
        dnc_fifo_single[i], const_opt[i], leaky_mass_1[i], leaky_mass_2_opt[
            i], exact_mass_2_opt[i] = regulated_comparison(
                aggregation=agg,
                sigma_single=sigma_single,
                rho_single=rho_single,
                service_rate=service_rate * agg,
                perform_param=perform_param)

    results_df = pd.DataFrame(
        {
            "DNCBound": dnc_fifo_single,
            "constBound": const_opt,
            "leakyMassOne": leaky_mass_1,
            "leakyMassTwo": leaky_mass_2_opt,
            "exactMassTwo": exact_mass_2_opt
        },
        index=aggregations)

    filename = "regulated_single_{0}_sigma_{1}_rho_{2}_utilization_{3}".format(
        perform_param.to_string(), str(sigma_single), str(rho_single),
        str(rho_single / service_rate))

    results_df.to_csv(
        filename + '.csv', index=True, quoting=csv.QUOTE_NONNUMERIC)

    return results_df


if __name__ == '__main__':
    DELAY6 = PerformParameter(
        perform_metric=PerformMetric.DELAY, value=10**(-6))

    NUMBER_AGGREGATIONS = [
        1, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350
    ]

    RHO_SINGLE = 0.1
    SERVICE_RATE = 0.12
    SIGMA_VALUES = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 18.0, 20.0]

    for SIGMA in SIGMA_VALUES:
        print(
            compare_aggregation(
                aggregations=NUMBER_AGGREGATIONS,
                sigma_single=SIGMA,
                rho_single=RHO_SINGLE,
                service_rate=SERVICE_RATE,
                perform_param=DELAY6))
