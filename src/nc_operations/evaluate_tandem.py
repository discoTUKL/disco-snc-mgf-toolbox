"""Helper function to evaluate a single hop."""

from math import exp, log
from typing import List

from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_operations.perform_enum import PerformEnum
from nc_operations.stability_check import stability_check
from nc_server.server import Server
from utils.helper_functions import get_p_n
from utils.perform_parameter import PerformParameter


def evaluate_tandem(foi: ArrivalDistribution,
                    ser_list: List[Server],
                    theta: float,
                    perform_param: PerformParameter,
                    p_list: List[float],
                    indep=True) -> float:
    if foi.is_discrete() is False:
        raise ValueError("Distribution must be discrete")

    if indep:
        p_list = [1.0]
    else:
        if len(p_list) != (len(ser_list) - 1):
            raise ValueError(f"number of p={len(p_list)} and length of "
                             f"ser_list={len(ser_list)} - 1 have to match")

        p_n = get_p_n(p_list=p_list, indep=False)

    sigma_sum = 0.0
    denominator = 1.0
    if indep:
        for server in ser_list:
            stability_check(arr=foi, ser=server, theta=theta, indep=True)
            sigma_sum += server.sigma(theta)
            denominator *= 1 - exp(theta *
                                   (foi.rho(theta) - server.rho(theta)))

        sigma_sum += foi.sigma(theta)

    else:
        for i in range(len(ser_list) - 1):
            stability_check(arr=foi,
                            ser=ser_list[i],
                            theta=theta,
                            indep=False,
                            p=1.0,
                            q=p_list[i])
            sigma_sum += ser_list[i].sigma(p_list[i] * theta)
            denominator *= 1 - exp(
                theta * (foi.rho(theta) - ser_list[i].rho(p_list[i] * theta)))

        sigma_sum += ser_list[-1].sigma(p_n * theta)
        sigma_sum += foi.sigma(theta)
        stability_check(arr=foi,
                        ser=ser_list[-1],
                        theta=theta,
                        indep=False,
                        p=1.0,
                        q=p_n)
        denominator *= 1 - exp(
            theta * (foi.rho(theta) - ser_list[-1].rho(p_n * theta)))

    second_factor = exp(theta * sigma_sum) / denominator

    if perform_param.perform_metric == PerformEnum.BACKLOG_PROB:
        return exp(-theta * perform_param.value) * second_factor

    elif perform_param.perform_metric == PerformEnum.BACKLOG:
        return log(second_factor / perform_param.value) / theta

    elif perform_param.perform_metric == PerformEnum.DELAY_PROB:
        if indep:
            return exp(-theta * ser_list[-1].rho(theta) *
                       perform_param.value) * second_factor
        else:
            return exp(-theta * ser_list[-1].rho(p_n * theta) *
                       perform_param.value) * second_factor

    elif perform_param.perform_metric == PerformEnum.DELAY:
        if indep:
            return log(second_factor /
                       perform_param.value) / (theta * ser_list[-1].rho(theta))
        else:
            return log(second_factor / perform_param.value) / (
                theta * ser_list[-1].rho(p_n * theta))

    elif perform_param.perform_metric == PerformEnum.OUTPUT:
        return exp(
            theta * foi.rho(theta) * perform_param.value) * second_factor

    else:
        raise NameError(f"{perform_param.perform_metric} is an infeasible "
                        f"performance metric")
