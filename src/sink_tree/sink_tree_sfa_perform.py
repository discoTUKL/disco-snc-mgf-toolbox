"""PMOO for sink tree"""

from typing import List

from utils.perform_parameter import PerformParameter
from utils.setting import Setting
from nc_operations.evaluate_single_hop import evaluate_single_hop
from nc_operations.operations import AggregateList, Convolve, Deconvolve, Leftover
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_service.service import Service
from nc_service.constant_rate_server import ConstantRate


class SinkTreeSFA(Setting):
    """Canonical tandem with SFA analysis"""

    def __init__(self, arr_list: List[ArrivalDistribution],
                 ser_list: List[ConstantRate],
                 perform_param: PerformParameter) -> None:
        # The first element in the arrival list in dedicated to the foi
        if len(arr_list) != (len(ser_list) + 1):
            raise ValueError(
                "number of arrivals {0} and servers {1} have to match".format(
                    len(arr_list), len(ser_list)))

        self.arr_list = arr_list
        self.ser_list = ser_list
        self.perform_param = perform_param
        self.number_servers = len(ser_list)

    def bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        if self.number_servers == 1:
            s_net: Service = Leftover(
                arr=self.arr_list[1], ser=self.ser_list[0])

        elif self.number_servers == 2:
            leftover_service_list = [
                Leftover(arr=self.arr_list[1], ser=self.ser_list[0]),
                AggregateList(arr_list=self.arr_list[2], p_list=[]),
                Deconvolve(arr=self.arr_list[1], ser=self.ser_list[0])
            ]

            s_net: Service = Convolve(
                ser1=leftover_service_list[0],
                ser2=leftover_service_list[1],
                indep=False,
                p=param_list[1])

        else:
            raise ValueError(
                "This number of servers {0} is not implemented yet".format(
                    self.number_servers))

        return evaluate_single_hop(
            foi=self.arr_list[0],
            s_net=s_net,
            theta=theta,
            perform_param=self.perform_param)

        # s_net: Service = Leftover(
        #     arr=self.arr_list[self.number_servers + 1],
        #     ser=self.ser_list[self.number_servers])
        #
        # for _i in range(self.number_servers, -1, -1):
        #     s_net = Convolve(ser1=s_net, ser2=self.ser_list[_i])
        #     s_net = Leftover(arr=self.arr_list[_i + 1], ser=s_net)
        #
        # return evaluate_single_hop(
        #     foi=self.arr_list[0],
        #     s_net=s_net,
        #     theta=theta,
        #     perform_param=self.perform_param)
