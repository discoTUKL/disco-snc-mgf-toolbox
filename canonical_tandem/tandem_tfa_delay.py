"""TFA hop-by-hop for canonical tree"""

from typing import List

from library.setting import Setting
from nc_operations.operations import Deconvolve, Leftover
from nc_operations.performance_bounds import delay
from nc_processes.arrival_distribution import ArrivalDistribution
from nc_processes.service import Service
from nc_processes.service_distribution import ConstantRate


class TandemTFADelay(Setting):
    """Canonical tandem with hop-by-hop analysis"""

    def __init__(self, arr_list: List[ArrivalDistribution],
                 ser_list: List[ConstantRate], prob_d: float) -> None:
        # The first element in the arrival list in dedicated to the foi
        if len(arr_list) != (len(ser_list) + 1):
            raise ValueError(
                "number of arrivals {0} and servers {1} have to match".format(
                    len(arr_list), len(ser_list)))

        self.arr_list = arr_list
        self.ser_list = ser_list
        self.prob_d = prob_d
        self.number_servers = len(ser_list)

    def bound(self, theta: float) -> float:
        leftover_service_list: List[Service] = [
            Leftover(arr=self.arr_list[i + 1], ser=self.ser_list[i])
            for i in range(self.number_servers)
        ]

        delay_val = 0.0

        input_traffic = self.arr_list[0]

        for i in range(self.number_servers):
            delay_val += delay(
                arr=input_traffic,
                ser=leftover_service_list[i],
                theta=theta,
                prob_d=self.prob_d)

            input_traffic = Deconvolve(
                arr=input_traffic, ser=leftover_service_list[i])

        return delay_val
