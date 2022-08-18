"""Helper function to evaluate a single hop."""

from nc_arrivals.arrival import Arrival
from nc_arrivals.regulated_arrivals import DetermTokenBucket
from nc_server.rate_latency_server import RateLatencyServer
from nc_server.server import Server
from utils.exceptions import IllegalArgumentError
from utils.perform_parameter import PerformParameter

from nc_operations.aggregate import AggregateHomogeneous
from nc_operations.dnc_delay import dnc_delay
from nc_operations.perform_enum import PerformEnum
from nc_operations.performance_bounds import (backlog, backlog_prob, delay,
                                              delay_prob, output)
from nc_operations.performance_bounds_geom import (backlog_geom,
                                                   backlog_prob_geom,
                                                   delay_geom, delay_prob_geom)


def single_hop_bound(foi: Arrival,
                     s_e2e: Server,
                     theta: float,
                     perform_param: PerformParameter,
                     indep=True,
                     p=1.0,
                     geom_series=False) -> float:
    if indep:
        p = 1.0

    match perform_param.perform_metric:
        case PerformEnum.BACKLOG_PROB:
            if geom_series:
                return backlog_prob_geom(arr=foi,
                                         ser=s_e2e,
                                         theta=theta,
                                         backlog_value=perform_param.value,
                                         indep=indep,
                                         p=p)

            else:
                return backlog_prob(arr=foi,
                                    ser=s_e2e,
                                    theta=theta,
                                    backlog_value=perform_param.value,
                                    indep=indep,
                                    p=p)

        case PerformEnum.BACKLOG:
            if geom_series:
                return backlog_geom(arr=foi,
                                    ser=s_e2e,
                                    theta=theta,
                                    prob_b=perform_param.value,
                                    indep=indep,
                                    p=p)

            else:
                return backlog(arr=foi,
                               ser=s_e2e,
                               theta=theta,
                               prob_b=perform_param.value,
                               indep=indep,
                               p=p)

        case PerformEnum.DELAY_PROB:
            if geom_series:
                return delay_prob_geom(arr=foi,
                                       ser=s_e2e,
                                       theta=theta,
                                       delay_value=perform_param.value,
                                       indep=indep,
                                       p=p)

            return delay_prob(arr=foi,
                              ser=s_e2e,
                              theta=theta,
                              delay_value=perform_param.value,
                              indep=indep,
                              p=p)

        case PerformEnum.DELAY:
            if isinstance(foi, DetermTokenBucket) and isinstance(
                    s_e2e, RateLatencyServer):
                return dnc_delay(tb=foi, rl=s_e2e)

            if geom_series:
                return delay_geom(arr=foi,
                                  ser=s_e2e,
                                  theta=theta,
                                  prob_d=perform_param.value,
                                  indep=indep,
                                  p=p)

            else:
                return delay(arr=foi,
                             ser=s_e2e,
                             theta=theta,
                             prob_d=perform_param.value,
                             indep=indep,
                             p=p)

        case PerformEnum.OUTPUT:
            return output(arr=foi,
                          ser=s_e2e,
                          theta=theta,
                          delta_time=perform_param.value,
                          indep=indep,
                          p=p)

        case _:
            raise IllegalArgumentError(f"{perform_param.perform_metric} is an "
                                       f"infeasible performance metric")


def single_hop_homog_agg(foi_arr_single: Arrival,
                         n: int,
                         s_e2e: Server,
                         theta: float,
                         perform_param: PerformParameter,
                         indep=True,
                         geom_series=True) -> float:
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
