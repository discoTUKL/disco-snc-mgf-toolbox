"""Evaluate a general tandem."""

from math import exp
from typing import List

import scipy.optimize

from nc_operations.flow import Flow
from nc_operations.perform_enum import PerformEnum
from nc_server.server import Server
from utils.exceptions import IllegalArgumentError, ParameterOutOfBounds
from utils.perform_parameter import PerformParameter


def pmoo_explicit(foi: Flow,
                  cross_flows: List[Flow],
                  ser_list: List[Server],
                  theta: float,
                  perform_param: PerformParameter,
                  indep=True) -> float:
    """Explicit computation"""

    if foi.arr.is_discrete() is False:
        raise NotImplementedError(
            "Only implemented for discrete-time processes")

    if (perform_param.perform_metric is not PerformEnum.DELAY_PROB
            and perform_param.perform_metric is not PerformEnum.DELAY):
        raise IllegalArgumentError(
            "This function can only be used for the delay / delay probability")

    if indep is False:
        raise NotImplementedError("Only implemented for independent processes")

    residual_rate_list = [0.0] * len(ser_list)
    sigma_sum = 0.0

    foi_rate = foi.arr.rho(theta=theta)

    # add all latencies
    for k, server in enumerate(ser_list):
        residual_rate_list[k] = server.rho(theta=theta)

        sigma_sum += server.sigma(theta=theta)

    # add all bursts and compute residual rates
    sigma_sum += foi.arr.sigma(theta=theta)

    for i, cross_flow in enumerate(cross_flows):
        sigma_sum += cross_flow.arr.sigma(theta=theta)
        for server_index in cross_flow.server_indices:
            cross_arr_rate = cross_flow.arr.rho(theta=theta)

            residual_rate_list[server_index] -= cross_arr_rate

    for res_rate in residual_rate_list:
        if res_rate - foi_rate <= 0:
            raise ParameterOutOfBounds("Stability condition is violated")

    if perform_param.perform_metric == PerformEnum.DELAY_PROB:
        sum_j = 0.0

        for j, residual_rate_j in enumerate(residual_rate_list):

            prod_k = 1.0

            for k, residual_rate_k in enumerate(residual_rate_list):
                if k is not j:
                    rate_diff = residual_rate_j - residual_rate_k
                    if rate_diff == 0.0:
                        raise IllegalArgumentError("Multiplicity is not = 1")

                    prod_k *= 1 / (1 - exp(theta * rate_diff))

            prod_k *= exp(
                theta * sigma_sum) / (1 - exp(theta *
                                              (foi_rate - residual_rate_j)))
            prod_k *= exp(-theta * residual_rate_j * perform_param.value)
            sum_j += prod_k

        return sum_j

    elif perform_param.perform_metric == PerformEnum.DELAY:
        target_delay_prob = perform_param.value

        def helper_function(delay: float) -> float:
            perform_delay = PerformParameter(
                perform_metric=PerformEnum.DELAY_PROB, value=delay)
            current_delay_prob = pmoo_explicit(foi=foi,
                                               cross_flows=cross_flows,
                                               ser_list=ser_list,
                                               theta=theta,
                                               perform_param=perform_delay,
                                               indep=indep)

            return target_delay_prob - current_delay_prob

        res = scipy.optimize.bisect(helper_function,
                                    a=0.1,
                                    b=10000,
                                    full_output=True)
        return res[0]

    else:
        raise IllegalArgumentError(
            f"{perform_param.perform_metric} is an infeasible "
            f"performance metric")
