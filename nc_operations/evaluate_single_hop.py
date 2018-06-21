"""Helper function to evaluate a single hop."""

from library.perform_parameter import PerformParameter
from nc_operations.perform_metric import PerformMetric
from nc_operations.performance_bounds import Delay, DelayProb, Output
from nc_operations.performance_bounds_discretized import (DelayDiscretized,
                                                          DelayProbDiscretized,
                                                          OutputDiscretized)
from nc_processes.arrival_distribution import ArrivalDistribution
from nc_processes.service import Service


def evaluate_single_hop(foi: ArrivalDistribution, s_net: Service,
                        theta: float, perform_param: PerformParameter) -> float:
    if perform_param.perform_metric == PerformMetric.DELAY_PROB:
        if foi.is_discrete():
            return DelayProb(
                arr=foi, ser=s_net).bound(
                theta=theta, delay=perform_param.value)
        else:
            return DelayProbDiscretized(
                arr=foi, ser=s_net).bound(
                theta=theta, delay=perform_param.value)

    elif perform_param.perform_metric == PerformMetric.DELAY:
        if foi.is_discrete():
            return Delay(
                arr=foi, ser=s_net).bound(
                theta=theta, prob_d=perform_param.value)
        else:
            return DelayDiscretized(
                arr=foi, ser=s_net).bound(
                theta=theta, prob_d=perform_param.value)

    elif perform_param.perform_metric == PerformMetric.OUTPUT:
        if foi.is_discrete():
            return Output(
                arr=foi, ser=s_net).bound(
                theta=theta, delta_time=perform_param.value)
        else:
            return OutputDiscretized(
                arr=foi, ser=s_net).bound(
                theta=theta, delta_time=perform_param.value)

    else:
        raise NameError("{0} is an infeasible performance metric".format(
            perform_param.perform_metric))
