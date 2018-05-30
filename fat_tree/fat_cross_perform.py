"""Fat tree topology."""

from typing import List

from library.perform_parameter import PerformParameter
from library.setting_new import SettingNew
from nc_operations.deconvolve_lya import DeconvolveLya
from nc_operations.operations import AggregateList, Deconvolve, Leftover
from nc_operations.perform_metric import PerformMetric
from nc_operations.performance_bounds import Delay, DelayProb
from nc_processes.arrival import Arrival
from nc_processes.arrival_distribution import ArrivalDistribution
from nc_processes.service import Service
from nc_processes.service_distribution import ServiceDistribution


class FatCrossPerform(SettingNew):
    """Fat tree cross topology with the Simple topology as a sub-problem."""

    def __init__(self, arr_list: List[ArrivalDistribution],
                 ser_list: List[ServiceDistribution],
                 perform_param: PerformParameter) -> None:
        # The first element in these lists in dedicated to the foi
        self.arr_list = arr_list
        self.ser_list = ser_list
        self.perform_param = perform_param

    def get_bound(self, theta: float) -> float:
        number_servers = len(self.arr_list)

        output_list: List[Arrival] = [
            Deconvolve(arr=self.arr_list[i], ser=self.ser_list[i])
            for i in range(1, number_servers)
        ]
        # we use i + 1, since i = 0 is the foi

        aggregated_cross: Arrival = AggregateList(arr_list=output_list)
        ser1_left: Service = Leftover(
            arr=aggregated_cross, ser=self.ser_list[0])

        if self.perform_param.perform_metric == PerformMetric.DELAY_PROB:
            fat_cross_delay_prob = DelayProb(
                arr=self.arr_list[0], ser=ser1_left)

            return fat_cross_delay_prob.bound(
                theta=theta, delay=self.perform_param.value)
        elif self.perform_param.perform_metric == PerformMetric.DELAY:
            fat_cross_delay = Delay(arr=self.arr_list[0], ser=ser1_left)

            return fat_cross_delay.bound(
                theta=theta, prob_d=self.perform_param.value)
        else:
            raise NameError(self.perform_param.perform_metric +
                            " is an infeasible performance metric")

    def get_new_bound(self, param_list: List[float]) -> float:
        if len(param_list) != len(self.arr_list):
            raise NameError("Check number of parameters")

        number_servers = len(self.arr_list)

        output_list: List[Arrival] = [
            DeconvolveLya(
                arr=self.arr_list[i],
                ser=self.ser_list[i],
                l_lya=param_list[i]) for i in range(1, number_servers)
        ]
        # we use i + 1, since i = 0 is the foi

        aggregated_cross: Arrival = AggregateList(arr_list=output_list)
        ser1_left: Service = Leftover(
            arr=aggregated_cross, ser=self.ser_list[0])

        if self.perform_param.perform_metric == PerformMetric.DELAY_PROB:
            fat_cross_new_delay_prob = DelayProb(
                arr=self.arr_list[0], ser=ser1_left)
            return fat_cross_new_delay_prob.bound(
                theta=param_list[0], delay=self.perform_param.value)
        elif self.perform_param.perform_metric == PerformMetric.DELAY:
            fat_cross_new_delay = Delay(arr=self.arr_list[0], ser=ser1_left)

            return fat_cross_new_delay.bound(
                theta=param_list[0], prob_d=self.perform_param.value)
        else:
            raise NameError(self.perform_param.perform_metric.name +
                            " is an infeasible performance metric")

    def to_string(self) -> str:
        for arr in self.arr_list:
            print(arr.to_string())
        for ser in self.ser_list:
            print(ser.to_string())
        return self.__class__.__name__ + "_" + self.perform_param.to_string()
