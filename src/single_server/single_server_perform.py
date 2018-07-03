"""Single server topology class"""

from typing import List
from warnings import warn

from library.perform_parameter import PerformParameter
from library.setting_new import SettingNew
from nc_operations.evaluate_single_hop import evaluate_single_hop
from nc_operations.perform_enum import PerformEnum
from nc_operations.performance_bounds_discretized import delay_prob_discretized
from nc_operations.performance_bounds_lya import (delay_prob_lya, output_lya,
                                                  output_lya_discretized)
from nc_processes.arrival_distribution import (ArrivalDistribution,
                                               ExponentialArrival)
from nc_processes.constant_rate_server import ConstantRate


class SingleServerPerform(SettingNew):
    """Single server topology class."""

    def __init__(self, arr: ArrivalDistribution, const_rate: ConstantRate,
                 perform_param: PerformParameter) -> None:
        """

        :param arr:           arrival process
        :param const_rate:           service
        :param perform_param: performance parameter
        """
        self.arr = arr
        self.ser = const_rate
        self.perform_param = perform_param

    def bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        return evaluate_single_hop(
            foi=self.arr,
            s_net=self.ser,
            theta=theta,
            perform_param=self.perform_param)

    def new_bound(self, param_l_list: List[float]) -> float:
        if self.perform_param.perform_metric == PerformEnum.DELAY_PROB:
            if self.arr.is_discrete():
                return delay_prob_lya(
                    arr=self.arr,
                    ser=self.ser,
                    theta=param_l_list[0],
                    delay=self.perform_param.value,
                    l_lya=param_l_list[1])
            else:
                warn("old approach is applied")
                return delay_prob_discretized(
                    arr=self.arr,
                    ser=self.ser,
                    theta=param_l_list[0],
                    delay_value=self.perform_param.value)

        elif self.perform_param.perform_metric == PerformEnum.OUTPUT:
            if self.arr.is_discrete():
                return output_lya(
                    arr=self.arr,
                    ser=self.ser,
                    theta=param_l_list[0],
                    delta_time=self.perform_param.value,
                    l_lya=param_l_list[1])
            else:
                return output_lya_discretized(
                    arr=self.arr,
                    ser=self.ser,
                    theta=param_l_list[0],
                    delta_time=self.perform_param.value,
                    l_lya=param_l_list[1])

        else:
            raise NameError("{0} is an infeasible performance metric".format(
                self.perform_param.perform_metric))

    def to_string(self) -> str:
        return self.__class__.__name__ + "_" + self.arr.to_value(
        ) + "_" + self.ser.to_value() + self.perform_param.to_name_value()


if __name__ == '__main__':
    EXP_ARRIVAL1 = ExponentialArrival(lamb=1.0)
    CONST_RATE16 = ConstantRate(rate=1.6)
    OUTPUT_4 = PerformParameter(perform_metric=PerformEnum.OUTPUT, value=4)
    EX_OUTPUT = SingleServerPerform(
        arr=EXP_ARRIVAL1, const_rate=CONST_RATE16, perform_param=OUTPUT_4)
    print(EX_OUTPUT.bound(param_list=[0.5]))
    print(EX_OUTPUT.new_bound(param_l_list=[0.5, 1.2]))

    DELAY_PROB_4 = PerformParameter(
        perform_metric=PerformEnum.DELAY_PROB, value=4)
    EX_DELAY_PROB = SingleServerPerform(
        arr=EXP_ARRIVAL1, const_rate=CONST_RATE16, perform_param=DELAY_PROB_4)
    print(EX_DELAY_PROB.bound(param_list=[0.5]))
    print(EX_DELAY_PROB.new_bound(param_l_list=[0.5, 1.2]))
