"""Single server topology class"""

from typing import List

from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.iid import DM1
from nc_operations.perform_enum import PerformEnum
from nc_operations.single_hop_bound import single_hop_bound
from nc_server.constant_rate_server import ConstantRateServer
from utils.perform_parameter import PerformParameter

from h_mitigator.performance_bounds_power_mit import (delay_prob_power_mit,
                                                      output_power_mit)
from h_mitigator.setting_mitigator import SettingMitigator


class SingleServerMitPerform(SettingMitigator):
    """Single server topology class."""
    def __init__(self,
                 arr_list: List[ArrivalDistribution],
                 server: ConstantRateServer,
                 perform_param: PerformParameter,
                 indep=True) -> None:
        """

        :param arr_list:           arrival process
        :param server:           server
        :param perform_param: performance parameter
        :param indep:        true if arrivals and service are independent
        """
        self.arr_list = arr_list
        self.server = server
        self.perform_param = perform_param
        self.indep = indep

    def standard_bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        if self.indep:
            p = 1.0
        else:
            p = param_list[1]

        return single_hop_bound(foi=self.arr_list[0],
                                s_e2e=self.server,
                                theta=theta,
                                perform_param=self.perform_param,
                                indep=self.indep,
                                p=p)

    def h_mit_bound(self, param_l_list: List[float]) -> float:
        if not self.indep:
            raise NotImplementedError

        match self.perform_param.perform_metric:
            case PerformEnum.OUTPUT:
                return output_power_mit(arr=self.arr_list[0],
                                        ser=self.server,
                                        theta=param_l_list[0],
                                        delta_time=self.perform_param.value,
                                        l_power=param_l_list[1])

            case PerformEnum.DELAY_PROB:
                return delay_prob_power_mit(arr=self.arr_list[0],
                                            ser=self.server,
                                            theta=param_l_list[0],
                                            delay=self.perform_param.value,
                                            l_power=param_l_list[1])

            case _:
                raise NotImplementedError(
                    f"{self.perform_param.perform_metric} is "
                    f"not implemented")

    def approximate_utilization(self) -> float:
        sum_average_rates = 0.0
        for arrival in self.arr_list:
            sum_average_rates += arrival.average_rate()

        return sum_average_rates / self.server.rate

    def to_string(self) -> str:
        return self.to_name() + "_" + self.arr_list[0].to_value(
        ) + "_" + self.server.to_value() + self.perform_param.__str__()


if __name__ == '__main__':
    EXP_ARRIVAL1 = [DM1(lamb=1.0)]
    CONST_RATE16 = ConstantRateServer(rate=1.6)
    OUTPUT_4 = PerformParameter(perform_metric=PerformEnum.OUTPUT, value=4)
    EX_OUTPUT = SingleServerMitPerform(arr_list=EXP_ARRIVAL1,
                                       server=CONST_RATE16,
                                       perform_param=OUTPUT_4)
    print(EX_OUTPUT.standard_bound(param_list=[0.5]))
    print(EX_OUTPUT.h_mit_bound(param_l_list=[0.5, 1.2]))

    DELAY_PROB_4 = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                    value=4)
    EX_DELAY_PROB = SingleServerMitPerform(arr_list=EXP_ARRIVAL1,
                                           server=CONST_RATE16,
                                           perform_param=DELAY_PROB_4)
    print(EX_DELAY_PROB.standard_bound(param_list=[0.5]))
    print(EX_DELAY_PROB.h_mit_bound(param_l_list=[0.5, 1.2]))
