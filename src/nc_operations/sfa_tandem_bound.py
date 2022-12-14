"""Evaluate a tandem of servers (residual service given)."""

import math
from typing import List

import numpy as np
import scipy.optimize
import scipy.special
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_server.server import Server
from utils.exceptions import IllegalArgumentError, ParameterOutOfBounds
from utils.helper_functions import get_p_n
from utils.perform_parameter import PerformParameter

from nc_operations.e2e_enum import E2EEnum
from nc_operations.perform_enum import PerformEnum
from nc_operations.stability_check import stability_check


def sfa_tandem_bound(foi: ArrivalDistribution,
                     leftover_service_list: List[Server],
                     theta: float,
                     perform_param: PerformParameter,
                     p_list: List[float],
                     e2e_enum: E2EEnum,
                     indep=True) -> float:
    if foi.is_discrete() is False:
        raise NotImplementedError("Only implemented for discrete-time processes")

    if indep:
        p_list = [1.0]
    else:
        if len(p_list) != (len(leftover_service_list) - 1):
            raise IllegalArgumentError(f"number of p={len(p_list)} and length of "
                                       f"ser_list={len(leftover_service_list)} - 1 have to match")

        if isinstance(p_list, np.ndarray):
            p_list = np.append(p_list, get_p_n(p_list=p_list))
        else:
            p_list.append(get_p_n(p_list=p_list))

    foi_rate = foi.rho(theta=theta)

    residual_rate_list = [0.0] * len(leftover_service_list)
    residual_rate_with_foi_list = [0.0] * len(leftover_service_list)
    sigma_sum = 0.0

    if indep:
        for i, server in enumerate(leftover_service_list):
            stability_check(arr=foi, ser=server, theta=theta, indep=True)

            sigma_sum += server.sigma(theta=theta)
            residual_rate_list[i] = server.rho(theta=theta)
            residual_rate_with_foi_list[i] = residual_rate_list[i] - foi_rate

    else:
        for i, server in enumerate(leftover_service_list):
            stability_check(arr=foi, ser=server, theta=theta, indep=False, p=1.0, q=p_list[i])

            # p = 1.0 is from the foi

            sigma_sum += server.sigma(theta=p_list[i] * theta)
            residual_rate_list[i] = server.rho(theta=p_list[i] * theta)
            residual_rate_with_foi_list[i] = residual_rate_list[i] - foi_rate

    sigma_sum += foi.sigma(theta=theta)

    if e2e_enum == E2EEnum.ARR_RATE:
        gamma = 1.0

        for residual_rate_with_foi in residual_rate_with_foi_list:
            gamma *= 1 / (1 - math.exp(-theta * residual_rate_with_foi))

        if perform_param.perform_metric == PerformEnum.BACKLOG_PROB:
            return math.exp(-theta * perform_param.value) * math.exp(theta * sigma_sum) * gamma

        elif perform_param.perform_metric == PerformEnum.BACKLOG:
            return (theta * sigma_sum + math.log(gamma / perform_param.value)) / theta

        elif perform_param.perform_metric == PerformEnum.DELAY_PROB:
            return math.exp(-theta * foi_rate * perform_param.value) * math.exp(theta * sigma_sum) * gamma

        elif perform_param.perform_metric == PerformEnum.DELAY:
            return (theta * sigma_sum + math.log(gamma / perform_param.value)) / (theta * foi_rate)

        elif perform_param.perform_metric == PerformEnum.OUTPUT:
            return math.exp(theta * foi_rate * perform_param.value) * math.exp(theta * sigma_sum) * gamma

        else:
            raise NotImplementedError(f"{perform_param.perform_metric} is an infeasible " f"performance metric")

    elif e2e_enum == E2EEnum.MIN_RATE:
        min_residual_rate = min(residual_rate_list)
        min_rate_with_foi = min_residual_rate - foi_rate

        q = math.exp(-theta * min_rate_with_foi)
        d_lower = len(leftover_service_list) * q / (1 - q)

        if perform_param.perform_metric == PerformEnum.DELAY_PROB:
            if perform_param.value >= d_lower:
                T_over_n = perform_param.value / len(leftover_service_list)
                zeta = (1 + T_over_n)**(1 + T_over_n) / (T_over_n**T_over_n)

                return math.exp(-theta * min_residual_rate * perform_param.value) * math.exp(
                    theta * sigma_sum) * (zeta**len(leftover_service_list))
            else:
                raise ParameterOutOfBounds("Zeta condition is violated")

        elif perform_param.perform_metric == PerformEnum.DELAY:
            T_over_n = d_lower / len(leftover_service_list)
            zeta = (1 + T_over_n)**(1 + T_over_n) / (T_over_n**T_over_n)

            delay_bound = (theta * sigma_sum + math.log(1 / perform_param.value) +
                           len(leftover_service_list) * math.log(zeta)) / (theta * min_residual_rate)

            if delay_bound >= d_lower:
                return delay_bound
            else:
                raise ParameterOutOfBounds("Zeta condition is violated")

        #     counter = 0
        #
        #     if delay_bound > d_lower + 1e-6:
        #         while delay_bound > d_lower + 1e-6 and counter < 5:
        #             d_lower = delay_bound
        #
        #             T_over_n = delay_bound / len(leftover_service_list)
        #             zeta = (1 + T_over_n)**(1 + T_over_n) / (T_over_n**
        #                                                      T_over_n)
        #
        #             delay_bound = sigma_sum / min_residual_rate + (
        #                 len(leftover_service_list) * math.log(zeta) -
        #                 math.log(perform_param.value)) / (theta * min_residual_rate)
        #
        #             counter += 1
        #
        #         return delay_bound
        #
        #     else:
        #         return d_lower
        #
        # else:
        #     raise NotImplementedError(f"{perform_param.perform_metric} is "
        #                               f"an infeasible performance metric")

    elif e2e_enum == E2EEnum.RATE_DIFF:
        min_residual_rate = min(residual_rate_list)
        dominating_pole_index = residual_rate_list.index(min(residual_rate_list))

        gamma = 1.0

        for index, residual_rate in enumerate(residual_rate_list):
            if index is not dominating_pole_index:
                rate_diff = residual_rate - min_residual_rate

                if rate_diff == 0.0:
                    delta = 0.5 * (min_residual_rate - foi_rate)
                    min_residual_rate -= delta
                    rate_diff = residual_rate - min_residual_rate

                try:
                    gamma *= 1 / (1 - math.exp(-theta * rate_diff))
                except ZeroDivisionError:
                    return math.inf

            min_residual_rate_with_foi = min_residual_rate - foi_rate
            gamma *= 1 / (1 - math.exp(-theta * min_residual_rate_with_foi))

        if perform_param.perform_metric == PerformEnum.DELAY_PROB:
            return math.exp(-theta * min_residual_rate * perform_param.value) * math.exp(theta * sigma_sum) * gamma

        elif perform_param.perform_metric == PerformEnum.DELAY:
            return (theta * sigma_sum + math.log(gamma / perform_param.value)) / (theta * min_residual_rate)

        else:
            raise NotImplementedError(f"{perform_param.perform_metric} is an infeasible " f"performance metric")

    elif e2e_enum == E2EEnum.ANALYTIC_COMBINATORICS:
        min_residual_rate = min(residual_rate_list)
        min_rate_with_foi = min_residual_rate - foi_rate

        gamma = 1.0
        k = 0

        for index, residual_rate in enumerate(residual_rate_list):
            rate_diff = residual_rate - min_residual_rate
            if rate_diff > 0:
                rate_diff = residual_rate - min_residual_rate
                gamma *= 1 / (1 - math.exp(-theta * rate_diff))
            else:
                k += 1

        factor = 0.0
        for i in range(k):
            factor += scipy.special.binom(perform_param.value + 1 + k - i - 2, perform_param.value + 1 - 1) / (
                (1 - math.exp(-theta * min_rate_with_foi))**(i + 1))

        if k == 0:
            factor = 1.0

        if perform_param.perform_metric == PerformEnum.DELAY_PROB:
            return math.exp(-theta * min_residual_rate * (perform_param.value + 1)) * math.exp(
                theta * foi_rate) * math.exp(theta * sigma_sum) * gamma * factor

        elif perform_param.perform_metric == PerformEnum.DELAY:
            target_delay_prob = perform_param.value

            def helper_function(delay: float) -> float:
                perform_delay = PerformParameter(perform_metric=PerformEnum.DELAY_PROB, value=delay)
                current_delay_prob = sfa_tandem_bound(foi=foi,
                                                      leftover_service_list=leftover_service_list,
                                                      theta=theta,
                                                      perform_param=perform_delay,
                                                      p_list=p_list,
                                                      e2e_enum=e2e_enum,
                                                      indep=indep)

                return target_delay_prob - current_delay_prob

            res = scipy.optimize.bisect(helper_function, a=1e-3, b=1e5, full_output=True)
            return res[0]

        else:
            raise NotImplementedError(f"{perform_param.perform_metric} is an infeasible " f"performance metric")

    else:
        raise NotImplementedError("SFA Analysis is not implemented")
