"""Fat tree topology."""

from typing import List

from h_mitigator.deconvolve_power_mit import DeconvolvePowerMit
from h_mitigator.setting_mitigator import SettingMitigator
from nc_arrivals.arrival import Arrival
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_operations.arb_scheduling import LeftoverARB
from nc_operations.single_hop_bound import single_hop_bound
from nc_operations.operations import AggregateList, Deconvolve
from nc_server.server import Server
from nc_server.server_distribution import ServerDistribution
from utils.perform_parameter import PerformParameter


class FatCrossPerform(SettingMitigator):
    """Fat tree cross topology with the Simple topology as a sub-problem."""
    def __init__(self, arr_list: List[ArrivalDistribution],
                 ser_list: List[ServerDistribution],
                 perform_param: PerformParameter) -> None:
        # The first element in these lists in dedicated to the foi
        if len(arr_list) != len(ser_list):
            raise ValueError(f"number of arrivals {len(arr_list)}"
                             f"and servers {len(ser_list)} have to match")
        self.arr_list = arr_list
        self.ser_list = ser_list
        self.perform_param = perform_param

        self.number_servers = len(ser_list)

    def standard_bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        output_list: List[Arrival] = [
            Deconvolve(arr=self.arr_list[i], ser=self.ser_list[i])
            for i in range(1, self.number_servers)
        ]
        # we use i + 1, since i = 0 is the foi

        aggregated_cross: Arrival = AggregateList(arr_list=output_list,
                                                  indep=True,
                                                  p_list=[])
        s_e2e: Server = LeftoverARB(ser=self.ser_list[0],
                                    cross_arr=aggregated_cross)

        return single_hop_bound(foi=self.arr_list[0],
                                s_e2e=s_e2e,
                                theta=theta,
                                perform_param=self.perform_param)

    def h_mit_bound(self, param_l_list: List[float]) -> float:
        output_list: List[Arrival] = [
            DeconvolvePowerMit(arr=self.arr_list[i],
                               ser=self.ser_list[i],
                               l_power=param_l_list[i])
            for i in range(1, self.number_servers)
        ]
        # we use i + 1, since i = 0 is the foi

        aggregated_cross: Arrival = AggregateList(arr_list=output_list,
                                                  indep=True,
                                                  p_list=[])
        s_e2e: Server = LeftoverARB(ser=self.ser_list[0],
                                    cross_arr=aggregated_cross)

        return single_hop_bound(foi=self.arr_list[0],
                                s_e2e=s_e2e,
                                theta=param_l_list[0],
                                perform_param=self.perform_param)

    def approximate_utilization(self) -> float:
        sum_average_rates = 0.0
        for arrival in self.arr_list:
            sum_average_rates += arrival.average_rate()

        # for i in range(1, self.number_servers):
        #     util_at_server_i = self.arr_list[i].average_rate(
        #     ) / self.ser_list[i].rate
        #     if util_at_server_i > max_util:
        #         max_util = util_at_server_i

        return sum_average_rates / self.ser_list[0].average_rate()

    def to_string(self) -> str:
        for arr in self.arr_list:
            print(arr.to_value())
        for ser in self.ser_list:
            print(ser.to_value())
        return self.to_name() + "_" + self.perform_param.__str__()
