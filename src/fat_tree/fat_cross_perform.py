"""Fat tree topology."""

from typing import List

from h_mitigator.deconvolve_power_mit import DeconvolvePowerMit
from h_mitigator.setting_mitigator import SettingMitigator
from nc_arrivals.arrival import Arrival
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_operations.evaluate_single_hop import evaluate_single_hop
from nc_operations.operations import AggregateList, Deconvolve, Leftover
from nc_service.constant_rate_server import ConstantRate
from nc_service.service import Service
from utils.perform_parameter import PerformParameter


class FatCrossPerform(SettingMitigator):
    """Fat tree cross topology with the Simple topology as a sub-problem."""

    def __init__(self, arr_list: List[ArrivalDistribution],
                 ser_list: List[ConstantRate],
                 perform_param: PerformParameter) -> None:
        # The first element in these lists in dedicated to the foi
        if len(arr_list) != len(ser_list):
            raise ValueError(f"number of arrivals {len(arr_list)}"
                             f"and servers {len(ser_list)} have to match")

        self.arr_list = arr_list
        self.ser_list = ser_list
        self.perform_param = perform_param
        self.number_servers = len(ser_list)

    def bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        output_list: List[Arrival] = [
            Deconvolve(arr=self.arr_list[i], ser=self.ser_list[i])
            for i in range(1, self.number_servers)
        ]
        # we use i + 1, since i = 0 is the foi

        aggregated_cross: Arrival = AggregateList(
            arr_list=output_list, p_list=[])
        s_net: Service = Leftover(ser=self.ser_list[0], arr=aggregated_cross)

        return evaluate_single_hop(
            foi=self.arr_list[0],
            s_net=s_net,
            theta=theta,
            perform_param=self.perform_param)

    def h_mit_bound(self, param_l_list: List[float]) -> float:
        # len(param_list) = theta (1) + output bounds (len(arr_list)-1)
        if len(param_l_list) != len(self.arr_list):
            raise NameError("Check number of parameters")

        output_list: List[Arrival] = [
            DeconvolvePowerMit(
                arr=self.arr_list[i],
                ser=self.ser_list[i],
                l_power=param_l_list[i])
            for i in range(1, self.number_servers)
        ]
        # we use i + 1, since i = 0 is the foi

        aggregated_cross: Arrival = AggregateList(
            arr_list=output_list, p_list=[])
        s_net: Service = Leftover(ser=self.ser_list[0], arr=aggregated_cross)

        return evaluate_single_hop(
            foi=self.arr_list[0],
            s_net=s_net,
            theta=param_l_list[0],
            perform_param=self.perform_param)

    def to_string(self) -> str:
        for arr in self.arr_list:
            print(arr.to_value())
        for ser in self.ser_list:
            print(ser.to_value())
        return self.to_name() + "_" + self.perform_param.to_name_value()
