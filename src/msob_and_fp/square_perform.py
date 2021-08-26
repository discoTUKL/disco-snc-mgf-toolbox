"""Splitting triangle network."""

from math import inf
from typing import List

from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.regulated_arrivals import DetermTokenBucket
from nc_operations.arb_scheduling import LeftoverARB
from nc_operations.operations import Convolve, Deconvolve
from nc_operations.single_hop_bound import single_hop_bound
from nc_server.constant_rate_server import ConstantRateServer
from utils.exceptions import ParameterOutOfBounds
from utils.perform_parameter import PerformParameter

from msob_and_fp.setting_avoid_dep import SettingMSOBFP


class SquarePerform(SettingMSOBFP):
    def __init__(self, arr_list: List[ArrivalDistribution],
                 ser_list: List[ConstantRateServer],
                 perform_param: PerformParameter) -> None:
        self.arr_list = arr_list
        self.ser_list = ser_list
        self.perform_param = perform_param

    def standard_bound(self, param_list: List[float]) -> float:
        theta = param_list[0]
        p = param_list[1]

        a_2 = self.arr_list[1]
        a_3 = self.arr_list[2]
        a_4 = self.arr_list[3]
        s_1 = self.ser_list[0]
        s_2 = self.ser_list[1]
        s_3 = self.ser_list[2]
        s_4 = self.ser_list[3]

        d_3_3 = Deconvolve(arr=a_3, ser=LeftoverARB(ser=s_3, cross_arr=a_2))
        d_4_4 = Deconvolve(arr=a_4,
                           ser=LeftoverARB(ser=s_4,
                                           cross_arr=Deconvolve(arr=a_2,
                                                                ser=s_3)))

        s_1_lo = LeftoverARB(ser=s_1, cross_arr=d_3_3)
        s_2_lo = LeftoverARB(ser=s_2, cross_arr=d_4_4)

        s_e2e = Convolve(ser1=s_1_lo, ser2=s_2_lo, indep=False, p=p)

        return single_hop_bound(foi=self.arr_list[0],
                                s_e2e=s_e2e,
                                theta=theta,
                                perform_param=self.perform_param,
                                indep=True)

    def server_bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        a_2 = self.arr_list[1]
        a_3 = self.arr_list[2]
        a_4 = self.arr_list[3]
        s_1 = self.ser_list[0]
        s_2 = self.ser_list[1]
        s_3 = self.ser_list[2]
        s_4 = self.ser_list[3]

        try:
            d_3_3 = DetermTokenBucket(sigma_single=0.0,
                                      rho_single=s_3.rate,
                                      n=1)
            d_4_4 = Deconvolve(arr=a_4,
                               ser=LeftoverARB(ser=s_4,
                                               cross_arr=Deconvolve(arr=a_2,
                                                                    ser=s_3)))

            s_1_lo = LeftoverARB(ser=s_1, cross_arr=d_3_3)
            s_2_lo = LeftoverARB(ser=s_2, cross_arr=d_4_4)

            s_net_1 = Convolve(ser1=s_1_lo, ser2=s_2_lo)

            res_1 = single_hop_bound(foi=self.arr_list[0],
                                     s_e2e=s_net_1,
                                     theta=theta,
                                     perform_param=self.perform_param)

        except ParameterOutOfBounds:
            res_1 = inf

        try:
            d_3_3 = Deconvolve(arr=a_3,
                               ser=LeftoverARB(ser=s_3, cross_arr=a_2))
            d_4_4 = DetermTokenBucket(sigma_single=0.0,
                                      rho_single=s_4.rate,
                                      n=1)

            s_1_lo = LeftoverARB(ser=s_1, cross_arr=d_3_3)
            s_2_lo = LeftoverARB(ser=s_2, cross_arr=d_4_4)

            s_net_2 = Convolve(ser1=s_1_lo, ser2=s_2_lo)

            res_2 = single_hop_bound(foi=self.arr_list[0],
                                     s_e2e=s_net_2,
                                     theta=theta,
                                     perform_param=self.perform_param)

        except ParameterOutOfBounds:
            res_2 = inf

        return min(res_1, res_2)

    def fp_bound(self, param_list: List[float]) -> float:
        theta = param_list[0]
        p = param_list[1]

        a_2 = self.arr_list[1]
        a_3 = self.arr_list[2]
        a_4 = self.arr_list[3]
        s_1 = self.ser_list[0]
        s_2 = self.ser_list[1]
        s_3 = self.ser_list[2]
        s_4 = self.ser_list[3]

        d_3_3 = Deconvolve(arr=a_3, ser=LeftoverARB(ser=s_3, cross_arr=a_2))
        d_4_4 = Deconvolve(arr=a_4,
                           ser=LeftoverARB(ser=s_4,
                                           cross_arr=Deconvolve(arr=a_2,
                                                                ser=s_3)))

        s_12_conv = Convolve(ser1=s_1,
                             ser2=LeftoverARB(ser=s_2, cross_arr=d_4_4))

        s_net = LeftoverARB(ser=s_12_conv, cross_arr=d_3_3, indep=False, p=p)

        return single_hop_bound(foi=self.arr_list[0],
                                s_e2e=s_net,
                                theta=theta,
                                perform_param=self.perform_param,
                                indep=True)

    def approximate_utilization(self) -> float:
        a_foi_rate = self.arr_list[0].average_rate()
        a_3_rate = self.arr_list[2].average_rate()
        a_4_rate = self.arr_list[3].average_rate()

        c_1 = self.ser_list[0].rate
        c_2 = self.ser_list[1].rate

        util_s_1 = (a_foi_rate + a_3_rate) / c_1
        util_s_2 = (a_foi_rate + a_4_rate) / c_2

        return max(util_s_1, util_s_2)

    def server_util(self, server_index: int) -> float:
        a_foi_rate = self.arr_list[0].average_rate()
        a_2_rate = self.arr_list[1].average_rate()
        a_3_rate = self.arr_list[2].average_rate()
        a_4_rate = self.arr_list[3].average_rate()

        c_1 = self.ser_list[0].rate
        c_2 = self.ser_list[1].rate
        c_3 = self.ser_list[2].rate
        c_4 = self.ser_list[3].rate

        if server_index is 0:
            return (a_foi_rate + a_3_rate) / c_1
        elif server_index is 1:
            return (a_foi_rate + a_4_rate) / c_2
        elif server_index is 2:
            return (a_2_rate + a_3_rate) / c_3
        elif server_index is 3:
            return (a_2_rate + a_4_rate) / c_4
        else:
            raise ValueError("Wrong server index")

    def to_string(self) -> str:
        for arr in self.arr_list:
            print(arr.to_value())
        for ser in self.ser_list:
            print(ser.to_value())
        return self.to_name() + "_" + self.perform_param.__str__()


if __name__ == '__main__':
    from nc_arrivals.iid import DM1
    from nc_operations.perform_enum import PerformEnum
    from nc_server.constant_rate_server import ConstantRateServer
    from optimization.optimize import Optimize

    from msob_and_fp.optimize_fp_bound import OptimizeFPBound
    from msob_and_fp.optimize_server_bound import OptimizeServerBound

    # from optimization.function_optimizer import optimizer_perform
    # DELAY_PROB_TIME = PerformParameter(
    #     perform_metric=PerformEnum.DELAY_PROB, value=6)
    # DELAY_PROB_TIME = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
    #                                    value=8)
    DELAY_PROB_TIME = PerformParameter(perform_metric=PerformEnum.DELAY,
                                       value=1E-3)

    ARR_LIST = [DM1(lamb=2.3), DM1(lamb=4.5), DM1(lamb=1.7), DM1(lamb=4.5)]

    SER_LIST = [
        ConstantRateServer(rate=1.2),
        ConstantRateServer(rate=6.2),
        ConstantRateServer(rate=7.3),
        ConstantRateServer(rate=6.2)
    ]

    if ARR_LIST[0].is_discrete() is False:
        raise ValueError("Distribution must be discrete")

    RANGES = [slice(0.1, 10.0, 0.1)]
    RANGES_2 = [slice(0.1, 10.0, 0.1), slice(1.1, 10.0, 0.1)]

    PRINT_X = True

    print("Utilization:")
    print(
        SquarePerform(arr_list=ARR_LIST,
                      ser_list=SER_LIST,
                      perform_param=DELAY_PROB_TIME).approximate_utilization())

    print("Standard Approach:")
    print(
        Optimize(setting=SquarePerform(arr_list=ARR_LIST,
                                       ser_list=SER_LIST,
                                       perform_param=DELAY_PROB_TIME),
                 number_param=2,
                 print_x=PRINT_X).grid_search(grid_bounds=[(0.1, 10.0),
                                                           (1.1, 10.0)],
                                              delta=0.1))

    print("Server Bound:")
    print(
        OptimizeServerBound(
            setting_msob_fp=SquarePerform(arr_list=ARR_LIST,
                                          ser_list=SER_LIST,
                                          perform_param=DELAY_PROB_TIME),
            number_param=1,
            print_x=PRINT_X).grid_search(grid_bounds=[(0.1, 10.0)], delta=0.1))

    print("Flow Prolongation:")
    print(
        OptimizeFPBound(setting_msob_fp=SquarePerform(
            arr_list=ARR_LIST,
            ser_list=SER_LIST,
            perform_param=DELAY_PROB_TIME),
                        number_param=2,
                        print_x=PRINT_X).grid_search(grid_bounds=[(0.1, 10.0),
                                                                  (1.1, 10.0)],
                                                     delta=0.1))
