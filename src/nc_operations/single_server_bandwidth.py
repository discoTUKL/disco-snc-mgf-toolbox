"""Single server topology class"""

from typing import List

from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_operations.single_hop_bound import single_hop_bound
from nc_server.server import Server
from utils.exceptions import IllegalArgumentError
from utils.perform_parameter import PerformParameter
from utils.setting import Setting


class SingleServerBandwidth(Setting):
    """Single server topology class."""
    def __init__(self,
                 arr_list: List[ArrivalDistribution],
                 s_e2e: Server,
                 perform_param: PerformParameter,
                 indep=True,
                 geom_series=True) -> None:
        """

        :param arr_list:           arrival process
        :param s_e2e:           e2e service process
        :param perform_param: performance parameter
        :param indep:        true if arrivals and service are independent
        :param geom_series: use geometric series or integral bound
        """
        self.arr_list = arr_list
        self.s_e2e = s_e2e
        self.perform_param = perform_param
        self.indep = indep
        self.geom_series = geom_series

    def standard_bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        if self.indep:
            p = 1.0
        else:
            p = param_list[1]

        return single_hop_bound(foi=self.arr_list[0],
                                s_e2e=self.s_e2e,
                                theta=theta,
                                perform_param=self.perform_param,
                                indep=self.indep,
                                p=p,
                                geom_series=self.geom_series)

    def approximate_utilization(self) -> float:
        raise IllegalArgumentError("this method cannot be called")
