"""Helper function to evaluate a single hop."""

from library.perform_parameter import PerformParameter
from nc_operations.perform_enum import PerformEnum
from nc_operations.performance_bounds import (backlog, backlog_prob, delay,
                                              delay_prob, output)
from nc_operations.performance_bounds_discretized import (
    backlog_discretized, backlog_prob_discretized, delay_discretized,
    delay_prob_discretized, output_discretized)
from nc_processes.arrival_distribution import ArrivalDistribution
from nc_processes.service import Service


def evaluate_single_hop(foi: ArrivalDistribution,
                        s_net: Service,
                        theta: float,
                        perform_param: PerformParameter,
                        indep=True,
                        p=1.0) -> float:
    if indep:
        p = 1.0
    else:
        p = p

    if perform_param.perform_metric == PerformEnum.BACKLOG_PROB:
        if foi.is_discrete():
            return backlog_prob(
                arr=foi,
                ser=s_net,
                theta=theta,
                backlog_value=perform_param.value,
                indep=indep,
                p=p)
        else:
            return backlog_prob_discretized(
                arr=foi,
                ser=s_net,
                theta=theta,
                backlog_value=perform_param.value,
                indep=indep,
                p=p)

    elif perform_param.perform_metric == PerformEnum.BACKLOG:
        if foi.is_discrete():
            return backlog(
                arr=foi,
                ser=s_net,
                theta=theta,
                prob_b=perform_param.value,
                indep=indep,
                p=p)
        else:
            return backlog_discretized(
                arr=foi,
                ser=s_net,
                theta=theta,
                prob_b=perform_param.value,
                indep=indep,
                p=p)

    elif perform_param.perform_metric == PerformEnum.DELAY_PROB:
        if foi.is_discrete():
            return delay_prob(
                arr=foi,
                ser=s_net,
                theta=theta,
                delay_value=perform_param.value,
                indep=indep,
                p=p)
        else:
            return delay_prob_discretized(
                arr=foi,
                ser=s_net,
                theta=theta,
                delay_value=perform_param.value,
                indep=indep,
                p=p)

    elif perform_param.perform_metric == PerformEnum.DELAY:
        if foi.is_discrete():
            return delay(
                arr=foi,
                ser=s_net,
                theta=theta,
                prob_d=perform_param.value,
                indep=indep,
                p=p)
        else:
            return delay_discretized(
                arr=foi,
                ser=s_net,
                theta=theta,
                prob_d=perform_param.value,
                indep=indep,
                p=p)

    elif perform_param.perform_metric == PerformEnum.OUTPUT:
        if foi.is_discrete():
            return output(
                arr=foi,
                ser=s_net,
                theta=theta,
                delta_time=perform_param.value,
                indep=indep,
                p=p)
        else:
            return output_discretized(
                arr=foi,
                ser=s_net,
                theta=theta,
                delta_time=perform_param.value,
                indep=indep,
                p=p)

    else:
        raise NameError("{0} is an infeasible performance metric".format(
            perform_param.perform_metric))
