"""Compare Delay for different bounds for regulated Arrivals"""

from typing import List, Tuple

from dnc.dnc_fifo_delay import DNCFIFODelay
from library.perform_parameter import PerformParameter
from nc_processes.arrival_distribution import (
    LeakyBucketMassOne, LeakyBucketMassTwo, LeakyBucketMassTwoExact,
    TokenBucketConstant)
from nc_processes.service import ConstantRate
from optimization.optimize import Optimize
from single_server.single_server_perform import SingleServerPerform


def regulated_comparison(aggregation: int, sigma_single: float,
                         rho_single: float, service_rate: float,
                         perform_param: PerformParameter) -> Tuple(float):
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
    leaky_mass_1_opt = Optimize(
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

    return dnc_fifo_single, const_opt, leaky_mass_1_opt, leaky_mass_2_opt, exact_mass_2_opt


def compare_aggregation(aggregations: List[int], sigma_single: float,
                        rho_single: float, service_rate: float,
                        perform_param: PerformParameter):
    for agg in aggregations:
        print(agg)
