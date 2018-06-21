"""Single server topology class"""

from typing import List

from library.perform_parameter import PerformParameter
from library.setting_new import SettingNew
from nc_operations.evaluate_single_hop import evaluate_single_hop
from nc_operations.perform_metric import PerformMetric
from nc_operations.performance_bounds_lya import (DelayProbLya, OutputLya,
                                                  OutputLyaDiscretized)
from nc_processes.arrival_distribution import (ArrivalDistribution,
                                               ExponentialArrival)
from nc_processes.service_distribution import ConstantRate, ServiceDistribution


class SingleServerPerform(SettingNew):
    """Single server topology class."""

    def __init__(self, arr: ArrivalDistribution, ser: ServiceDistribution,
                 perform_param: PerformParameter) -> None:
        """

        :param arr:           arrival process
        :param ser:           service
        :param perform_param: performance parameter
        """
        self.arr = arr
        self.ser = ser
        self.perform_param = perform_param

    def get_bound(self, theta: float) -> float:
        foi = self.arr
        s_net = self.ser

        return evaluate_single_hop(
            foi=foi,
            s_net=s_net,
            theta=theta,
            perform_param=self.perform_param)

    def get_new_bound(self, param_list: List[float]) -> float:
        if self.perform_param.perform_metric == PerformMetric.DELAY_PROB:
            if self.arr.is_discrete():
                return DelayProbLya(
                    arr=self.arr, ser=self.ser, l_lya=param_list[1]).bound(
                        theta=param_list[0], delay=self.perform_param.value)

        elif self.perform_param.perform_metric == PerformMetric.OUTPUT:
            if self.arr.is_discrete():
                return OutputLya(
                    arr=self.arr, ser=self.ser, l_lya=param_list[1]).bound(
                        theta=param_list[0],
                        delta_time=self.perform_param.value)
            else:
                return OutputLyaDiscretized(
                    arr=self.arr, ser=self.ser, l_lya=param_list[1]).bound(
                        theta=param_list[0],
                        delta_time=self.perform_param.value)

        else:
            raise NameError("{0} is an infeasible performance metric".format(
                self.perform_param.perform_metric))

    def to_string(self) -> str:
        return self.__class__.__name__ + "_" + self.arr.to_string(
        ) + "_" + self.ser.to_string() + self.perform_param.to_string()


if __name__ == '__main__':
    EXP_ARRIVAL1 = ExponentialArrival(lamb=1.0)
    CONST_RATE16 = ConstantRate(rate=1.6)
    OUTPUT_4 = PerformParameter(perform_metric=PerformMetric.OUTPUT, value=4)
    EX_OUTPUT = SingleServerPerform(
        arr=EXP_ARRIVAL1, ser=CONST_RATE16, perform_param=OUTPUT_4)
    print(EX_OUTPUT.get_bound(0.5))
    print(EX_OUTPUT.get_new_bound([0.5, 1.2]))

    DELAY_PROB_4 = PerformParameter(
        perform_metric=PerformMetric.DELAY_PROB, value=4)
    EX_DELAY_PROB = SingleServerPerform(
        arr=EXP_ARRIVAL1, ser=CONST_RATE16, perform_param=DELAY_PROB_4)
    print(EX_DELAY_PROB.get_bound(0.5))
    print(EX_DELAY_PROB.get_new_bound([0.5, 1.2]))
