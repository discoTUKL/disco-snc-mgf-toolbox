"""Single server topology class"""

from typing import List

from h_mitigator.performance_bounds_power_mit import (delay_prob_power_mit,
                                                      output_power_mit)
from h_mitigator.setting_mitigator import SettingMitigator
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.qt import DM1
from nc_operations.evaluate_single_hop import evaluate_single_hop
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from nc_server.server_distribution import ServerDistribution
from utils.perform_parameter import PerformParameter


class SingleServerMitPerform(SettingMitigator):
    """Single server topology class."""
    def __init__(self,
                 arr_list: List[ArrivalDistribution],
                 ser_list: List[ServerDistribution],
                 perform_param: PerformParameter,
                 indep=True) -> None:
        """

        :param arr_list:           arrival process
        :param ser_list:           service
        :param perform_param: performance parameter
        :param indep:        true if arrivals and service are independent
        """
        super().__init__(arr_list=arr_list,
                         ser_list=ser_list,
                         perform_param=perform_param)
        self.indep = indep

    def standard_bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        if self.indep:
            p = 1.0
        else:
            p = param_list[1]

        return evaluate_single_hop(foi=self.arr_list[0],
                                   s_e2e=self.ser_list[0],
                                   theta=theta,
                                   perform_param=self.perform_param,
                                   indep=self.indep,
                                   p=p)

    def h_mit_bound(self, param_l_list: List[float]) -> float:
        if not self.indep:
            raise NotImplementedError

        if self.perform_param.perform_metric == PerformEnum.OUTPUT:
            return output_power_mit(arr=self.arr_list[0],
                                    ser=self.ser_list[0],
                                    theta=param_l_list[0],
                                    delta_time=self.perform_param.value,
                                    l_power=param_l_list[1])

        elif self.perform_param.perform_metric == PerformEnum.DELAY_PROB:
            return delay_prob_power_mit(arr=self.arr_list[0],
                                        ser=self.ser_list[0],
                                        theta=param_l_list[0],
                                        delay=self.perform_param.value,
                                        l_power=param_l_list[1])

        else:
            raise NotImplementedError(
                f"{self.perform_param.perform_metric} is "
                f"not implemented")

    def approximate_utilization(self) -> float:
        return self.arr_list.average_rate() / self.ser_list.average_rate()

    def to_string(self) -> str:
        return self.to_name() + "_" + self.arr_list.to_value(
        ) + "_" + self.ser_list.to_value() + self.perform_param.__str__()


if __name__ == '__main__':
    EXP_ARRIVAL1 = [DM1(lamb=1.0)]
    CONST_RATE16 = [ConstantRateServer(rate=1.6)]
    OUTPUT_4 = PerformParameter(perform_metric=PerformEnum.OUTPUT, value=4)
    EX_OUTPUT = SingleServerMitPerform(arr_list=EXP_ARRIVAL1,
                                       ser_list=CONST_RATE16,
                                       perform_param=OUTPUT_4)
    print(EX_OUTPUT.standard_bound(param_list=[0.5]))
    print(EX_OUTPUT.h_mit_bound(param_l_list=[0.5, 1.2]))

    DELAY_PROB_4 = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                    value=4)
    EX_DELAY_PROB = SingleServerMitPerform(arr_list=EXP_ARRIVAL1,
                                           ser_list=CONST_RATE16,
                                           perform_param=DELAY_PROB_4)
    print(EX_DELAY_PROB.standard_bound(param_list=[0.5]))
    print(EX_DELAY_PROB.h_mit_bound(param_l_list=[0.5, 1.2]))
