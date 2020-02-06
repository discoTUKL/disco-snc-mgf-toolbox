"""Helper function to evaluate a single hop."""

from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_operations.perform_enum import PerformEnum
from nc_operations.performance_bounds import (backlog, backlog_prob, delay,
                                              delay_prob, output)
from nc_server.server import Server
from utils.perform_parameter import PerformParameter


def evaluate_single_hop(foi: ArrivalDistribution,
                        s_e2e: Server,
                        theta: float,
                        perform_param: PerformParameter,
                        indep=True,
                        p=1.0,
                        use_standard=True) -> float:
    if indep:
        p = 1.0

    if perform_param.perform_metric == PerformEnum.BACKLOG_PROB:
        return backlog_prob(arr=foi,
                            ser=s_e2e,
                            theta=theta,
                            backlog_value=perform_param.value,
                            indep=indep,
                            p=p,
                            use_standard=use_standard)

    elif perform_param.perform_metric == PerformEnum.BACKLOG:
        return backlog(arr=foi,
                       ser=s_e2e,
                       theta=theta,
                       prob_b=perform_param.value,
                       indep=indep,
                       p=p,
                       use_standard=use_standard)

    elif perform_param.perform_metric == PerformEnum.DELAY_PROB:
        return delay_prob(arr=foi,
                          ser=s_e2e,
                          theta=theta,
                          delay_value=perform_param.value,
                          indep=indep,
                          p=p,
                          use_standard=use_standard)

    elif perform_param.perform_metric == PerformEnum.DELAY:
        return delay(arr=foi,
                     ser=s_e2e,
                     theta=theta,
                     prob_d=perform_param.value,
                     indep=indep,
                     p=p,
                     use_standard=use_standard)

    elif perform_param.perform_metric == PerformEnum.OUTPUT:
        return output(arr=foi,
                      ser=s_e2e,
                      theta=theta,
                      delta_time=perform_param.value,
                      indep=indep,
                      p=p)

    else:
        raise NameError(f"{perform_param.perform_metric} is an infeasible "
                        f"performance metric")
