"""Evaluate a general tandem."""

from math import exp, log
from typing import List

import scipy.optimize
import scipy.special
from nc_server.constant_rate_server import ConstantRateServer
from nc_server.server import Server
from utils.exceptions import ParameterOutOfBounds
from utils.perform_parameter import PerformParameter

from nc_operations.e2e_enum import E2EEnum
from nc_operations.flow import Flow
from nc_operations.perform_enum import PerformEnum


def pmoo_tandem_bound(foi: Flow,
                      cross_flows_on_foi_path: List[Flow],
                      ser_on_foi_path: List[Server],
                      theta: float,
                      perform_param: PerformParameter,
                      e2e_enum: E2EEnum,
                      cross_flows_not_on_foi_path=None,
                      ser_not_on_foi_path=None,
                      indep=True,
                      geom_series=True) -> float:
    """Pay bursts with foi arrival rate
    :param foi:            flow of interest
    :param cross_flows_on_foi_path:    cross flows on the foi's path
    :param ser_on_foi_path:       list of servers on the foi's path
    :param theta:           optimization parameter
    :param perform_param:  performance parameter
    :param e2e_enum:       Enum for e2e analysis types
    :param cross_flows_not_on_foi_path cross flows that are not
                            on the foi's path
    :param ser_not_on_foi_path list of servers that are not on the foi's path
    :param indep:          assumption of independent flows
    :param geom_series:     use of geometric series or integral bound
    :return:                bound
    """

    if foi.arr.is_discrete() is False:
        raise NotImplementedError(
            "Only implemented for discrete-time processes")

    if indep is False:
        raise NotImplementedError("Only implemented for independent processes")

    residual_rate_with_foi_list = [0.0] * len(ser_on_foi_path)
    residual_rate_list = [0.0] * len(ser_on_foi_path)
    sigma_sum = 0.0

    foi_rate = foi.arr.rho(theta=theta)

    for k, server in enumerate(ser_on_foi_path):
        residual_rate_list[k] = server.rho(theta=theta)
        residual_rate_with_foi_list[k] = residual_rate_list[k] - foi_rate
        sigma_sum += server.sigma(theta=theta)

    sigma_sum += foi.arr.sigma(theta=theta)

    for cross_flow in cross_flows_on_foi_path:
        sigma_sum += cross_flow.arr.sigma(theta=theta)

        for server_index in cross_flow.server_indices:
            cross_arr_rate = cross_flow.arr.rho(theta=theta)
            residual_rate_list[server_index] -= cross_arr_rate
            residual_rate_with_foi_list[server_index] -= cross_arr_rate

    for res_rate_with_foi in residual_rate_with_foi_list:
        if res_rate_with_foi <= 0:
            raise ParameterOutOfBounds("Stability condition is violated")

    if ser_not_on_foi_path is not None:
        residual_rate_not_on_foi_list = [0.0] * len(ser_not_on_foi_path)

        for j, server in enumerate(ser_not_on_foi_path):
            residual_rate_not_on_foi_list[j] = server.rho(theta=theta)
            sigma_sum += server.sigma(theta=theta)

        for cross_flow in cross_flows_not_on_foi_path:
            sigma_sum += cross_flow.arr.sigma(theta=theta)

            for server_index in cross_flow.server_indices:
                cross_arr_rate = cross_flow.arr.rho(theta=theta)
                residual_rate_not_on_foi_list[server_index] -= cross_arr_rate

        for res_rate_not_foi in residual_rate_not_on_foi_list:
            if res_rate_not_foi <= 0:
                raise ParameterOutOfBounds("Stability condition is violated")

            factor_not_on_foi = 1.0 / (1 - exp(-theta * res_rate_not_foi))

    if e2e_enum == E2EEnum.ARR_RATE:
        gamma = 1.0 * factor_not_on_foi

        for residual_rate_with_foi in residual_rate_with_foi_list:
            if geom_series:
                gamma *= 1 / (1 - exp(-theta * residual_rate_with_foi))
            else:
                gamma *= 1 / residual_rate_with_foi

        if perform_param.perform_metric == PerformEnum.BACKLOG_PROB:
            return exp(-theta * perform_param.value) * exp(
                theta * sigma_sum) * gamma

        elif perform_param.perform_metric == PerformEnum.BACKLOG:
            return (theta * sigma_sum +
                    log(gamma / perform_param.value)) / theta

        elif perform_param.perform_metric == PerformEnum.DELAY_PROB:
            return exp(-theta * foi_rate * perform_param.value) * exp(
                theta * sigma_sum) * gamma

        elif perform_param.perform_metric == PerformEnum.DELAY:
            return (theta * sigma_sum +
                    log(gamma / perform_param.value)) / (theta * foi_rate)

        elif perform_param.perform_metric == PerformEnum.OUTPUT:
            return exp(theta * foi_rate * perform_param.value) * exp(
                theta * sigma_sum) * gamma

        else:
            raise NotImplementedError(
                f"{perform_param.perform_metric} is an infeasible "
                f"performance metric")

    elif e2e_enum == E2EEnum.MIN_RATE:
        min_residual_rate = min(residual_rate_list)
        min_residual_rate_with_foi = min_residual_rate - foi_rate

        q = exp(-theta * min_residual_rate_with_foi)
        d_lower = len(ser_on_foi_path) * q / (1 - q)

        if perform_param.perform_metric == PerformEnum.DELAY_PROB:
            if perform_param.value >= d_lower:
                T_over_n = perform_param.value / len(ser_on_foi_path)
                zeta = (1 + T_over_n)**(1 + T_over_n) / (T_over_n**T_over_n)

                return exp(-theta * min_residual_rate *
                           perform_param.value) * exp(theta * sigma_sum) * (
                               zeta**len(ser_on_foi_path)) * factor_not_on_foi
            else:
                raise ParameterOutOfBounds("Zeta condition is violated")

        elif perform_param.perform_metric == PerformEnum.DELAY:
            T_over_n = d_lower / len(ser_on_foi_path)
            # print(f"T_over_n={T_over_n}")
            zeta = (1 + T_over_n)**(1 + T_over_n) / (T_over_n**T_over_n)
            # print(f"zeta={zeta}")

            delay_bound = (theta * sigma_sum + log(1 / perform_param.value) +
                           len(ser_on_foi_path) * log(zeta) +
                           log(factor_not_on_foi)) / (theta *
                                                      min_residual_rate)

            return max(d_lower, delay_bound)

            # counter = 0
            #
            # if delay_bound > d_lower + 1e-6:
            #     while delay_bound > d_lower + 1e-6 and counter < 5:
            #         d_lower = delay_bound
            #
            #         T_over_n = delay_bound / len(ser_on_foi_path)
            #         zeta = (1 + T_over_n)**(1 + T_over_n) / (T_over_n**
            #                                                  T_over_n)
            #
            #         delay_bound = sigma_sum / min_residual_rate + (
            #             len(ser_on_foi_path) * log(zeta) +
            #             log(factor_not_on_foi) -
            #             log(perform_param.value)) / (theta * min_residual_rate)
            #
            #         counter += 1
            #
            #     return delay_bound
            #
            # else:
            #     return d_lower

        else:
            raise NotImplementedError("This function can only be used "
                                      "for the delay / delay probability")

    elif e2e_enum == E2EEnum.RATE_DIFF:
        min_residual_rate = min(residual_rate_list)
        dominating_pole_index = residual_rate_list.index(
            min(residual_rate_list))

        gamma = 1.0

        for index, residual_rate in enumerate(residual_rate_list):
            if index is not dominating_pole_index:
                rate_diff = residual_rate - min_residual_rate
                # print(f"rate_diff = {rate_diff}")

                if rate_diff == 0.0:
                    delta = 0.5 * (min_residual_rate - foi_rate)
                    min_residual_rate -= delta
                    rate_diff = residual_rate - min_residual_rate

                gamma *= 1 / (1 - exp(-theta * rate_diff))

        min_residual_rate_with_foi = min_residual_rate - foi_rate
        gamma *= 1 / (1 - exp(-theta * min_residual_rate_with_foi))
        gamma *= factor_not_on_foi

        if perform_param.perform_metric == PerformEnum.DELAY_PROB:
            return exp(-theta * min_residual_rate * perform_param.value) * exp(
                theta * sigma_sum) * gamma

        elif perform_param.perform_metric == PerformEnum.DELAY:
            return (theta * sigma_sum + log(gamma / perform_param.value)) / (
                theta * min_residual_rate)

        else:
            raise NotImplementedError(
                f"{perform_param.perform_metric} is an infeasible "
                f"performance metric")

    elif e2e_enum == E2EEnum.ANALYTIC_COMBINATORICS:
        min_residual_rate = min(residual_rate_list)
        min_residual_rate_with_foi = min_residual_rate - foi_rate

        gamma = 1.0
        k = 0

        for index, residual_rate in enumerate(residual_rate_list):
            rate_diff = residual_rate - min_residual_rate
            if rate_diff > 0:
                rate_diff = residual_rate - min_residual_rate
                gamma *= 1 / (1 - exp(-theta * rate_diff))
            else:
                k += 1

        factor = 0.0
        for i in range(k):
            factor += scipy.special.binom(
                perform_param.value + k - i - 2, perform_param.value - 1) / (
                    (1 - exp(-theta * min_residual_rate_with_foi))**(i + 1))

        gamma *= factor_not_on_foi

        if perform_param.perform_metric == PerformEnum.DELAY_PROB:
            return exp(-theta * min_residual_rate * perform_param.value) * exp(
                theta * sigma_sum) * gamma * factor

        elif perform_param.perform_metric == PerformEnum.DELAY:
            target_delay_prob = perform_param.value

            def helper_function(delay: float) -> float:
                perform_delay = PerformParameter(
                    perform_metric=PerformEnum.DELAY_PROB, value=delay)
                current_delay_prob = pmoo_tandem_bound(
                    foi=foi,
                    cross_flows_on_foi_path=cross_flows_on_foi_path,
                    ser_on_foi_path=ser_on_foi_path,
                    theta=theta,
                    perform_param=perform_delay,
                    e2e_enum=e2e_enum,
                    indep=indep,
                    geom_series=True)

                return target_delay_prob - current_delay_prob

            res = scipy.optimize.bisect(helper_function,
                                        a=1e-6,
                                        b=1e5,
                                        full_output=True)
            return res[0]

        else:
            raise NotImplementedError(
                f"{perform_param.perform_metric} is an infeasible "
                f"performance metric")

    else:
        raise NotImplementedError("PMOO Analysis is not implemented")


def pmoo_tandem_bound_mixed(foi: Flow,
                            cross_flows: List[Flow],
                            ser_list: List[ConstantRateServer],
                            theta: float,
                            perform_param: PerformParameter,
                            indep=True) -> float:
    """Pay bursts with min residual rate, approach as in Fidler06
    :param foi:            flow of interest
    :param cross_flows:    cross flows
    :param ser_list:       list of servers along the foi's path
    :param theta:           optimization parameter
    :param perform_param:  performance parameter
    :param indep:          assumption of independent flows
    :return:                bound
    """

    if foi.arr.is_discrete() is False:
        raise NotImplementedError(
            "Only implemented for discrete-time processes")

    if (perform_param.perform_metric is not PerformEnum.DELAY_PROB
            and perform_param.perform_metric is not PerformEnum.DELAY):
        raise NotImplementedError(
            "This function can only be used for the delay / delay probability")

    if indep is False:
        raise NotImplementedError("Only implemented for independent processes")

    residual_rate_list = [0.0] * len(ser_list)

    foi_rate = foi.arr.rho(theta=theta)

    for k, server in enumerate(ser_list):
        residual_rate_list[k] = server.rho(theta=theta)

    burst_cross_sum = 0.0

    for i, cross_flow in enumerate(cross_flows):
        burst_cross_sum += cross_flows[i].arr.sigma(theta=theta)
        for server_index in cross_flow.server_indices:
            cross_arr_rate = cross_flow.arr.rho(theta=theta)
            residual_rate_list[server_index] -= cross_arr_rate
            if residual_rate_list[server_index] - foi_rate <= 0:
                raise ParameterOutOfBounds("Stability condition is violated")

    min_residual_rate = min(residual_rate_list)

    min_residual_rate_with_foi = min_residual_rate - foi_rate
    gamma = 1 / (1 - exp(-theta * min_residual_rate_with_foi))

    # if perform_param.perform_metric == PerformEnum.BACKLOG_PROB:
    #     return exp(-theta * perform_param.value) * exp(
    #         theta * sigma_sum) * gamma
    #
    # elif perform_param.perform_metric == PerformEnum.BACKLOG:
    #     return (theta * sigma_sum + log(gamma / perform_param.value)) / theta

    # elif perform_param.perform_metric == PerformEnum.DELAY_PROB:
    #     return exp(-theta * foi_rate * perform_param.value) * exp(
    #         theta * sigma_sum) * gamma

    if perform_param.perform_metric == PerformEnum.DELAY:
        return (theta * foi.arr.sigma(theta=theta) + len(ser_list) * log(gamma)
                - log(perform_param.value)) / (theta * foi.arr.rho(
                    theta=theta)) + burst_cross_sum / min_residual_rate

    # elif perform_param.perform_metric == PerformEnum.OUTPUT:
    #     return exp(theta * foi.arr.rho(theta=theta) *
    #                perform_param.value) * exp(theta * sigma_sum) * gamma

    else:
        raise NotImplementedError(
            f"{perform_param.perform_metric} is an infeasible "
            f"performance metric")
