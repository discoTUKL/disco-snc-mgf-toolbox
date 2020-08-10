"""Overlapping (non-nested) tandem network."""

from typing import List

from msob_and_fp.setting_avoid_dep import SettingMSOBFP
from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.regulated_arrivals import DetermTokenBucket
from nc_operations.arb_scheduling import LeftoverARB
from nc_operations.operations import AggregateTwo, Convolve, Deconvolve
from nc_operations.single_hop_bound import single_hop_bound
from nc_server.constant_rate_server import ConstantRateServer
from utils.perform_parameter import PerformParameter


class OverlappingTandemPerform(SettingMSOBFP):
    def __init__(self, arr_list: List[ArrivalDistribution],
                 ser_list: List[ConstantRateServer],
                 perform_param: PerformParameter) -> None:
        self.arr_list = arr_list
        self.ser_list = ser_list
        self.perform_param = perform_param

    def standard_bound(self, param_list: List[float]) -> float:
        # conducts a PMOO analysis -> case distinction necessary
        theta = param_list[0]
        p = param_list[1]

        foi = self.arr_list[0]
        a_2 = self.arr_list[1]
        a_3 = self.arr_list[2]

        s_1 = self.ser_list[0]
        s_2 = self.ser_list[1]
        s_3 = self.ser_list[2]

        conv_s1_s2_lo = LeftoverARB(ser=Convolve(ser1=s_1,
                                                 ser2=LeftoverARB(
                                                     ser=s_2, cross_arr=a_3)),
                                    cross_arr=a_2)
        d_3_2 = Deconvolve(arr=a_3, ser=s_2)
        s3_lo = LeftoverARB(ser=s_3, cross_arr=d_3_2)

        s_e2e_1 = Convolve(ser1=conv_s1_s2_lo, ser2=s3_lo, indep=False, p=p)

        res_1 = single_hop_bound(foi=foi,
                                 s_e2e=s_e2e_1,
                                 theta=theta,
                                 perform_param=self.perform_param,
                                 indep=True)

        d_2_1 = Deconvolve(arr=a_2, ser=s_1)
        conv_s2_s3_lo = LeftoverARB(ser=Convolve(ser1=LeftoverARB(
            ser=s_2, cross_arr=d_2_1),
                                                 ser2=s_3),
                                    cross_arr=a_3)
        s1_lo = LeftoverARB(ser=s_1, cross_arr=a_2)

        s_e2e_2 = Convolve(ser1=s1_lo, ser2=conv_s2_s3_lo, indep=False, p=p)

        res_2 = single_hop_bound(foi=foi,
                                 s_e2e=s_e2e_2,
                                 theta=theta,
                                 perform_param=self.perform_param,
                                 indep=True)

        return min(res_1, res_2)

    def server_bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        foi = self.arr_list[0]
        a_2 = self.arr_list[1]
        a_3 = self.arr_list[2]

        s_1 = self.ser_list[0]
        s_2 = self.ser_list[1]
        s_3 = self.ser_list[2]

        conv_s1_s2_lo = LeftoverARB(ser=Convolve(ser1=s_1,
                                                 ser2=LeftoverARB(
                                                     ser=s_2, cross_arr=a_3)),
                                    cross_arr=a_2)
        d_3_2 = DetermTokenBucket(sigma_single=0.0, rho_single=s_2.rate, n=1)
        s3_lo = LeftoverARB(ser=s_3, cross_arr=d_3_2)

        s_e2e_1 = Convolve(ser1=conv_s1_s2_lo, ser2=s3_lo)

        res_1 = single_hop_bound(foi=foi,
                                 s_e2e=s_e2e_1,
                                 theta=theta,
                                 perform_param=self.perform_param,
                                 indep=True)

        d_2_1 = DetermTokenBucket(sigma_single=0.0, rho_single=s_1.rate, n=1)
        conv_s2_s3_lo = LeftoverARB(ser=Convolve(ser1=LeftoverARB(
            ser=s_2, cross_arr=d_2_1),
                                                 ser2=s_3),
                                    cross_arr=a_3)
        s1_lo = LeftoverARB(ser=s_1, cross_arr=a_2)

        s_e2e_2 = Convolve(ser1=s1_lo, ser2=conv_s2_s3_lo)

        res_2 = single_hop_bound(foi=self.arr_list[0],
                                 s_e2e=s_e2e_2,
                                 theta=theta,
                                 perform_param=self.perform_param,
                                 indep=True)

        return min(res_1, res_2)

    def fp_bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        foi = self.arr_list[0]
        a_2 = self.arr_list[1]
        a_3 = self.arr_list[2]

        s_1 = self.ser_list[0]
        s_2 = self.ser_list[1]
        s_3 = self.ser_list[2]

        s_23_conv = Convolve(ser1=s_2, ser2=s_3)
        s_23_lo = LeftoverARB(ser=s_23_conv, cross_arr=a_3)
        s_123_conv = Convolve(ser1=s_1, ser2=s_23_lo)
        s_e2e = LeftoverARB(ser=s_123_conv, cross_arr=a_2)

        return single_hop_bound(foi=foi,
                                s_e2e=s_e2e,
                                theta=theta,
                                perform_param=self.perform_param,
                                indep=True)

    def approximate_utilization(self) -> float:
        foi_rate = self.arr_list[0].average_rate()
        a_2_rate = self.arr_list[1].average_rate()
        a_3_rate = self.arr_list[2].average_rate()

        c_1 = self.ser_list[0].rate
        c_2 = self.ser_list[1].rate
        c_3 = self.ser_list[2].rate

        util_s_1 = (foi_rate + a_2_rate) / c_1
        util_s_2 = (foi_rate + a_2_rate + a_3_rate) / c_2
        util_s_3 = (foi_rate + a_3_rate) / c_3

        return max(util_s_1, util_s_2, util_s_3)

    def server_util(self, server_index: int) -> float:
        a_foi_rate = self.arr_list[0].average_rate()
        a_2_rate = self.arr_list[1].average_rate()
        a_3_rate = self.arr_list[2].average_rate()

        c_1 = self.ser_list[0].rate
        c_2 = self.ser_list[1].rate
        c_3 = self.ser_list[2].rate

        if server_index is 0:
            return (a_foi_rate + a_2_rate) / c_1
        elif server_index is 1:
            return (a_foi_rate + a_2_rate + a_3_rate) / c_2
        elif server_index is 2:
            return (a_foi_rate + a_3_rate) / c_3
        else:
            raise ValueError("Wrong server index")

    def to_string(self) -> str:
        for arr in self.arr_list:
            print(arr.to_value())
        for ser in self.ser_list:
            print(ser.to_value())
        return self.to_name() + "_" + self.perform_param.__str__()


def sfa_bound_old(param_list: [float, float,
                               float], arr_list: List[ArrivalDistribution],
                  ser_list: List[ConstantRateServer],
                  perform_param: PerformParameter) -> float:
    theta = param_list[0]

    foi = arr_list[0]
    a_2 = arr_list[1]
    a_3 = arr_list[2]

    s_1 = ser_list[0]
    s_2 = ser_list[1]
    s_3 = ser_list[2]

    s1_lo = LeftoverARB(ser=s_1, cross_arr=a_2)

    d_2_1 = Deconvolve(arr=a_2, ser=s_1)
    s2_lo = LeftoverARB(ser=s_2, cross_arr=AggregateTwo(arr1=d_2_1, arr2=a_3))

    d_3_2 = Deconvolve(arr=a_3, ser=s_2)
    # d_3_2 = Deconvolve(arr=a_3, ser=Leftover(ser=s_2, arr=a_2))
    s3_lo = LeftoverARB(ser=s_3, cross_arr=d_3_2)

    s_23_lo = Convolve(ser1=s2_lo, ser2=s3_lo, indep=False, p=param_list[1])

    s_e2e = Convolve(ser1=s1_lo, ser2=s_23_lo, indep=False, p=param_list[2])

    return single_hop_bound(foi=foi,
                            s_e2e=s_e2e,
                            theta=theta,
                            perform_param=perform_param,
                            indep=True)


if __name__ == '__main__':
    # from msob_and_fp.optimize_fp_bound import OptimizeFPBound
    # from msob_and_fp.optimize_server_bound import OptimizeServerBound
    # from nc_arrivals.qt import DM1
    # from nc_arrivals.qt import DPoisson1
    from nc_arrivals.markov_modulated import MMOODisc
    # from nc_arrivals.regulated_arrivals import LeakyBucketMassOne
    from nc_operations.perform_enum import PerformEnum
    from nc_server.constant_rate_server import ConstantRateServer
    from optimization.optimize import Optimize
    from optimization.function_optimizer import optimizer_perform

    # DELAY_PROB_TIME = PerformParameter(
    #     perform_metric=PerformEnum.DELAY_PROB, value=6)
    # DELAY_PROB_TIME = PerformParameter(perform_metric=PerformEnum.DELAY_PROB,
    #                                    value=8)

    DELAY_PROB_TIME = PerformParameter(perform_metric=PerformEnum.DELAY,
                                       value=10**(-6))
    # DELAY_PROB_TIME = PerformParameter(perform_metric=PerformEnum.OUTPUT,
    #                                    value=3)

    # ARR_LIST = [DM1(lamb=2.3), DM1(lamb=4.5), DM1(lamb=1.7)]

    # ARR_LIST = [DPoisson1(lamb=0.5), DPoisson1(lamb=0.2), DPoisson1(lamb=0.3)]
    ARR_LIST = [
        MMOODisc(stay_on=0.6, stay_off=0.4, peak_rate=0.9),
        MMOODisc(stay_on=0.6, stay_off=0.4, peak_rate=0.5),
        MMOODisc(stay_on=0.6, stay_off=0.4, peak_rate=0.7)
    ]

    # ARR_LIST = [LeakyBucketMassOne(sigma_single=1.0, rho_single=0.5),
    #             LeakyBucketMassOne(sigma_single=1.0, rho_single=0.2),
    #             LeakyBucketMassOne(sigma_single=1.0, rho_single=0.3)]

    # ARR_LIST = [
    #     TokenBucketConstant(sigma_single=1.0, rho_single=0.5),
    #     TokenBucketConstant(sigma_single=1.0, rho_single=0.2),
    #     TokenBucketConstant(sigma_single=1.0, rho_single=0.3)
    # ]

    SER_LIST = [
        ConstantRateServer(rate=1.2),
        ConstantRateServer(rate=6.2),
        ConstantRateServer(rate=7.3)
    ]

    if ARR_LIST[0].is_discrete() is False:
        raise ValueError("Distribution must be discrete")

    RANGES_1 = [slice(0.1, 10.0, 0.1)]
    RANGES_2 = [slice(0.1, 10.0, 0.1), slice(1.1, 10.0, 0.1)]
    RANGES_3 = [
        slice(0.1, 10.0, 0.1),
        slice(1.1, 10.0, 0.1),
        slice(1.1, 10.0, 0.1)
    ]

    RANGES_GPS = [
        slice(0.1, 10.0, 0.1),
        slice(0.1, 0.9, 0.05),
        slice(0.1, 0.9, 0.05),
        slice(0.1, 0.9, 0.05),
        slice(0.1, 0.9, 0.05)
    ]

    PRINT_X = True

    print("Utilization:")
    print(
        OverlappingTandemPerform(
            arr_list=ARR_LIST,
            ser_list=SER_LIST,
            perform_param=DELAY_PROB_TIME).approximate_utilization())

    print("Standard Approach with PMOO:")
    print(
        Optimize(setting=OverlappingTandemPerform(
            arr_list=ARR_LIST,
            ser_list=SER_LIST,
            perform_param=DELAY_PROB_TIME),
                 number_param=2,
                 print_x=PRINT_X).grid_search(bound_list=[(0.1, 10.0),
                                                          (1.1, 10.0)],
                                              delta=0.1))

    print("Standard Approach with SFA old:")
    print(
        optimizer_perform(fun=sfa_bound_old,
                          arr_list=ARR_LIST,
                          ser_list=SER_LIST,
                          perform_param=DELAY_PROB_TIME,
                          ranges=RANGES_3,
                          print_x=PRINT_X))

    # print("Server Bound:")
    # print(
    #     OptimizeServerBound(setting_avoid_dep=OverlappingTandemPerform(
    #         arr_list=ARR_LIST,
    #         ser_list=SER_LIST,
    #         perform_param=DELAY_PROB_TIME),
    #                         print_x=PRINT_X).grid_search(bound_list=[(0.1,
    #                                                                   10.0)],
    #                                                      delta=0.1))
    #
    # print("Flow Prolongation:")
    # print(
    #     OptimizeFPBound(setting_avoid_dep=OverlappingTandemPerform(
    #         arr_list=ARR_LIST,
    #         ser_list=SER_LIST,
    #         perform_param=DELAY_PROB_TIME),
    #                     print_x=PRINT_X).grid_search(bound_list=[(0.1, 10.0)],
    #                                                  delta=0.1))
