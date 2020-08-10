"""Helper function to evaluate a single hop."""

from nc_arrivals.arrival import Arrival
from nc_operations.operations import AggregateHomogeneous
from nc_operations.perform_enum import PerformEnum
from nc_operations.performance_bounds import (backlog, backlog_prob, delay,
                                              delay_prob, output)
from nc_server.server import Server
from utils.perform_parameter import PerformParameter


def single_hop_bound(foi: Arrival,
                     s_e2e: Server,
                     theta: float,
                     perform_param: PerformParameter,
                     indep=True,
                     p=1.0,
                     geom_series=True) -> float:
    if indep:
        p = 1.0

    if perform_param.perform_metric == PerformEnum.BACKLOG_PROB:
        return backlog_prob(arr=foi,
                            ser=s_e2e,
                            theta=theta,
                            backlog_value=perform_param.value,
                            indep=indep,
                            p=p,
                            geom_series=geom_series)

    elif perform_param.perform_metric == PerformEnum.BACKLOG:
        return backlog(arr=foi,
                       ser=s_e2e,
                       theta=theta,
                       prob_b=perform_param.value,
                       indep=indep,
                       p=p,
                       geom_series=geom_series)

    elif perform_param.perform_metric == PerformEnum.DELAY_PROB:
        return delay_prob(arr=foi,
                          ser=s_e2e,
                          theta=theta,
                          delay_value=perform_param.value,
                          indep=indep,
                          p=p,
                          geom_series=geom_series)

    elif perform_param.perform_metric == PerformEnum.DELAY:
        return delay(arr=foi,
                     ser=s_e2e,
                     theta=theta,
                     prob_d=perform_param.value,
                     indep=indep,
                     p=p,
                     geom_series=geom_series)

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


def single_hop_homog_agg(foi_arr_single: Arrival,
                         n: int,
                         s_e2e: Server,
                         theta: float,
                         perform_param: PerformParameter,
                         indep=True,
                         geom_series=True):
    if not indep:
        raise NotImplementedError

    if n > 1:
        return single_hop_bound(foi=AggregateHomogeneous(arr=foi_arr_single,
                                                         n=n,
                                                         indep=indep),
                                s_e2e=s_e2e,
                                theta=theta,
                                perform_param=perform_param,
                                geom_series=geom_series)
    else:
        return single_hop_bound(foi=foi_arr_single,
                                s_e2e=s_e2e,
                                theta=theta,
                                perform_param=perform_param,
                                geom_series=geom_series)
