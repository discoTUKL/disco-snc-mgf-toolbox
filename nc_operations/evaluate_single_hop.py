"""Helper function to evaluate a single hop."""

from library.perform_parameter import PerformParameter
from nc_operations.perform_enum import PerformEnum
from nc_operations.performance_bounds import delay, delay_prob, output
from nc_operations.performance_bounds_discretized import (
    delay_discretized, delay_prob_discretized, output_discretized)
from nc_processes.arrival_distribution import ArrivalDistribution
from nc_processes.service import Service


def evaluate_single_hop(foi: ArrivalDistribution, s_net: Service, theta: float,
                        perform_param: PerformParameter) -> float:
    if perform_param.perform_metric == PerformEnum.DELAY_PROB:
        if foi.is_discrete():
            return delay_prob(
                arr=foi,
                ser=s_net,
                theta=theta,
                delay_value=perform_param.value)
        else:
            return delay_prob_discretized(
                arr=foi, ser=s_net, theta=theta, delay=perform_param.value)

    elif perform_param.perform_metric == PerformEnum.DELAY:
        if foi.is_discrete():
            return delay(
                arr=foi, ser=s_net, theta=theta, prob_d=perform_param.value)
        else:
            return delay_discretized(
                arr=foi, ser=s_net, theta=theta, prob_d=perform_param.value)

    elif perform_param.perform_metric == PerformEnum.OUTPUT:
        if foi.is_discrete():
            return output(
                arr=foi,
                ser=s_net,
                theta=theta,
                delta_time=perform_param.value)
        else:
            return output_discretized(
                arr=foi,
                ser=s_net,
                theta=theta,
                delta_time=perform_param.value)

    else:
        raise NameError("{0} is an infeasible performance metric".format(
            perform_param.perform_metric))
