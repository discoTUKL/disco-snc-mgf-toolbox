"""Single server topology class"""

from typing import List

from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.qt import DM1
from nc_operations.evaluate_single_hop import evaluate_single_hop
from nc_operations.perform_enum import PerformEnum
from nc_server.constant_rate_server import ConstantRateServer
from utils.perform_parameter import PerformParameter
from utils.setting import Setting


class SingleServerPerform(Setting):
    """Single server topology class."""

    def __init__(self,
                 arr: ArrivalDistribution,
                 const_rate: ConstantRateServer,
                 perform_param: PerformParameter,
                 indep=True,
                 use_standard=True) -> None:
        """

        :param arr:           arrival process
        :param const_rate:           service
        :param perform_param: performance parameter
        :param indep:        true if arrivals and service are independent
        :param use_standard: type of bound
        """
        self.arr = arr
        self.const_rate = const_rate
        self.perform_param = perform_param
        self.indep = indep
        self.use_standard = use_standard

    def bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        if self.indep:
            p = 1.0
        else:
            p = param_list[1]

        return evaluate_single_hop(foi=self.arr,
                                   s_net=self.const_rate,
                                   theta=theta,
                                   perform_param=self.perform_param,
                                   indep=self.indep,
                                   p=p,
                                   use_standard=self.use_standard)

    def approximate_utilization(self) -> float:
        return self.arr.average_rate() / self.const_rate.rate

    def parameters_to_opt(self) -> int:
        return 1

    def to_string(self) -> str:
        return self.to_name() + "_" + self.arr.to_value(
        ) + "_" + self.const_rate.to_value() + self.perform_param.__str__()


if __name__ == '__main__':
    EXP_ARRIVAL1 = DM1(lamb=1.0)
    CONST_RATE16 = ConstantRateServer(rate=1.6)
    OUTPUT_4 = PerformParameter(perform_metric=PerformEnum.OUTPUT, value=4)
    EX_OUTPUT = SingleServerPerform(arr=EXP_ARRIVAL1,
                                    const_rate=CONST_RATE16,
                                    perform_param=OUTPUT_4)
    print(EX_OUTPUT.bound(param_list=[0.5]))

    DELAY_PROB_4 = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
                                    value=4)
    EX_DELAY_PROB = SingleServerPerform(arr=EXP_ARRIVAL1,
                                        const_rate=CONST_RATE16,
                                        perform_param=DELAY_PROB_4)
    print(EX_DELAY_PROB.bound(param_list=[0.5]))
