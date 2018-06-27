"""Fat tree topology."""

from typing import List

from library.perform_parameter import PerformParameter
from library.setting_new import SettingNew
from nc_operations.deconvolve_lya import DeconvolveLya
from nc_operations.evaluate_single_hop import evaluate_single_hop
from nc_operations.operations import AggregateList, Deconvolve, Leftover
from nc_processes.arrival import Arrival
from nc_processes.arrival_distribution import ArrivalDistribution
from nc_processes.service import Service
from nc_processes.service_distribution import ConstantRate


class FatCrossPerform(SettingNew):
    """Fat tree cross topology with the Simple topology as a sub-problem."""

    def __init__(self, arr_list: List[ArrivalDistribution],
                 ser_list: List[ConstantRate],
                 perform_param: PerformParameter) -> None:
        # The first element in these lists in dedicated to the foi
        if len(arr_list) != len(ser_list):
            raise ValueError(
                "number of arrivals {0} and servers {1} have to match".format(
                    len(arr_list), len(ser_list)))

        self.arr_list = arr_list
        self.ser_list = ser_list
        self.perform_param = perform_param
        self.number_servers = len(ser_list)

    def bound(self, theta: float) -> float:
        output_list: List[Arrival] = [
            Deconvolve(arr=self.arr_list[i], ser=self.ser_list[i])
            for i in range(1, self.number_servers)
        ]
        # we use i + 1, since i = 0 is the foi

        aggregated_cross: Arrival = AggregateList(arr_list=output_list)
        s_net: Service = Leftover(arr=aggregated_cross, ser=self.ser_list[0])

        return evaluate_single_hop(
            foi=self.arr_list[0],
            s_net=s_net,
            theta=theta,
            perform_param=self.perform_param)

    def new_bound(self, param_list: List[float]) -> float:
        if len(param_list) != len(self.arr_list):
            raise NameError("Check number of parameters")

        output_list: List[Arrival] = [
            DeconvolveLya(
                arr=self.arr_list[i],
                ser=self.ser_list[i],
                l_lya=param_list[i]) for i in range(1, self.number_servers)
        ]
        # we use i + 1, since i = 0 is the foi

        aggregated_cross: Arrival = AggregateList(arr_list=output_list)
        s_net: Service = Leftover(arr=aggregated_cross, ser=self.ser_list[0])

        return evaluate_single_hop(
            foi=self.arr_list[0],
            s_net=s_net,
            theta=param_list[0],
            perform_param=self.perform_param)

    def to_string(self) -> str:
        for arr in self.arr_list:
            print(arr.to_string())
        for ser in self.ser_list:
            print(ser.to_string())
        return self.__class__.__name__ + "_" + self.perform_param.to_string()
