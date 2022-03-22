"""Overlapping (non-nested) tandem network."""

from math import inf
from typing import List

from nc_arrivals.arrival_distribution import ArrivalDistribution
from nc_arrivals.regulated_arrivals import DetermTokenBucket
from nc_operations.aggregate import AggregateTwo
from nc_operations.arb_scheduling import LeftoverARB
from nc_operations.convolve import Convolve
from nc_operations.deconvolve import Deconvolve
from nc_operations.e2e_enum import E2EEnum
from nc_operations.gps_scheduling import LeftoverGPSPG
from nc_operations.sfa_tandem_bound import sfa_tandem_bound
from nc_operations.single_hop_bound import single_hop_bound
from nc_server.constant_rate_server import ConstantRateServer
from utils.exceptions import IllegalArgumentError
from utils.perform_parameter import PerformParameter
from utils.setting_sfa import SettingSFA

from msob_and_fp.setting_msob_fp import SettingMSOBFP


class OverlappingTandem(SettingMSOBFP):
    """Tandem with priorities f_1 <= f_2 <= f_3"""
    def __init__(self, arr_list: List[ArrivalDistribution],
                 ser_list: List[ConstantRateServer],
                 perform_param: PerformParameter) -> None:
        self.arr_list = arr_list
        self.ser_list = ser_list
        self.perform_param = perform_param

    def standard_bound(self, param_list: List[float]) -> float:
        """conducts a PMOO analysis -> case distinction necessary"""
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
        d_3_2 = DetermTokenBucket(sigma_single=0.0, rho_single=s_2.rate, m=1)
        s3_lo = LeftoverARB(ser=s_3, cross_arr=d_3_2)

        s_e2e_1 = Convolve(ser1=conv_s1_s2_lo, ser2=s3_lo)

        res_1 = single_hop_bound(foi=foi,
                                 s_e2e=s_e2e_1,
                                 theta=theta,
                                 perform_param=self.perform_param,
                                 indep=True)

        d_2_1 = DetermTokenBucket(sigma_single=0.0, rho_single=s_1.rate, m=1)
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

        if server_index == 0:
            return (a_foi_rate + a_2_rate) / c_1
        elif server_index == 1:
            return (a_foi_rate + a_2_rate + a_3_rate) / c_2
        elif server_index == 2:
            return (a_foi_rate + a_3_rate) / c_3
        else:
            raise IllegalArgumentError("Wrong server index")

    def to_string(self) -> str:
        for arr in self.arr_list:
            print(arr.to_value())
        for ser in self.ser_list:
            print(ser.to_value())
        return self.to_name() + "_" + self.perform_param.__str__()


class OverlappingTandemSFAPerform(SettingSFA):
    def __init__(self, arr_list: List[ArrivalDistribution],
                 ser_list: List[ConstantRateServer],
                 perform_param: PerformParameter) -> None:
        self.arr_list = arr_list
        self.ser_list = ser_list
        self.perform_param = perform_param

    def standard_bound(self, param_list: List[float]) -> float:
        """conducts an SFA analysis"""
        theta = param_list[0]

        foi = self.arr_list[0]
        a_2 = self.arr_list[1]
        a_3 = self.arr_list[2]

        s_1 = self.ser_list[0]
        s_2 = self.ser_list[1]
        s_3 = self.ser_list[2]

        s1_lo = LeftoverARB(ser=s_1, cross_arr=a_2)

        d_2_1 = Deconvolve(arr=a_2, ser=s_1)
        s2_lo = LeftoverARB(ser=s_2,
                            cross_arr=AggregateTwo(arr1=d_2_1, arr2=a_3))

        d_3_2 = Deconvolve(arr=a_3, ser=s_2)
        # d_3_2 = Deconvolve(arr=a_3, ser=Leftover(ser=s_2, arr=a_2))
        s3_lo = LeftoverARB(ser=s_3, cross_arr=d_3_2)

        s_23_lo = Convolve(ser1=s2_lo,
                           ser2=s3_lo,
                           indep=False,
                           p=param_list[1])

        s_e2e = Convolve(ser1=s1_lo,
                         ser2=s_23_lo,
                         indep=False,
                         p=param_list[2])

        return single_hop_bound(foi=foi,
                                s_e2e=s_e2e,
                                theta=theta,
                                perform_param=self.perform_param,
                                indep=True)

    def sfa_arr_bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        foi = self.arr_list[0]
        a_2 = self.arr_list[1]
        a_3 = self.arr_list[2]

        s_1 = self.ser_list[0]
        s_2 = self.ser_list[1]
        s_3 = self.ser_list[2]

        s1_lo = LeftoverARB(ser=s_1, cross_arr=a_2)

        d_2_1 = Deconvolve(arr=a_2, ser=s_1)
        s2_lo = LeftoverARB(ser=s_2,
                            cross_arr=AggregateTwo(arr1=d_2_1, arr2=a_3))

        d_3_2 = Deconvolve(arr=a_3, ser=s_2)
        # d_3_2 = Deconvolve(arr=a_3, ser=Leftover(ser=s_2, arr=a_2))
        s3_lo = LeftoverARB(ser=s_3, cross_arr=d_3_2)

        return sfa_tandem_bound(foi=foi,
                                leftover_service_list=[s1_lo, s2_lo, s3_lo],
                                theta=theta,
                                perform_param=self.perform_param,
                                p_list=param_list[1:],
                                e2e_enum=E2EEnum.ARR_RATE,
                                indep=False)

    def sfa_min_bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        foi = self.arr_list[0]
        a_2 = self.arr_list[1]
        a_3 = self.arr_list[2]

        s_1 = self.ser_list[0]
        s_2 = self.ser_list[1]
        s_3 = self.ser_list[2]

        s1_lo = LeftoverARB(ser=s_1, cross_arr=a_2)

        d_2_1 = Deconvolve(arr=a_2, ser=s_1)
        s2_lo = LeftoverARB(ser=s_2,
                            cross_arr=AggregateTwo(arr1=d_2_1, arr2=a_3))

        d_3_2 = Deconvolve(arr=a_3, ser=s_2)
        # d_3_2 = Deconvolve(arr=a_3, ser=Leftover(ser=s_2, arr=a_2))
        s3_lo = LeftoverARB(ser=s_3, cross_arr=d_3_2)

        return sfa_tandem_bound(foi=foi,
                                leftover_service_list=[s1_lo, s2_lo, s3_lo],
                                theta=theta,
                                perform_param=self.perform_param,
                                p_list=param_list[1:],
                                e2e_enum=E2EEnum.MIN_RATE,
                                indep=False)

    def sfa_rate_diff_bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        foi = self.arr_list[0]
        a_2 = self.arr_list[1]
        a_3 = self.arr_list[2]

        s_1 = self.ser_list[0]
        s_2 = self.ser_list[1]
        s_3 = self.ser_list[2]

        s1_lo = LeftoverARB(ser=s_1, cross_arr=a_2)

        d_2_1 = Deconvolve(arr=a_2, ser=s_1)
        s2_lo = LeftoverARB(ser=s_2,
                            cross_arr=AggregateTwo(arr1=d_2_1, arr2=a_3))

        d_3_2 = Deconvolve(arr=a_3, ser=s_2)
        # d_3_2 = Deconvolve(arr=a_3, ser=Leftover(ser=s_2, arr=a_2))
        s3_lo = LeftoverARB(ser=s_3, cross_arr=d_3_2)

        return sfa_tandem_bound(foi=foi,
                                leftover_service_list=[s1_lo, s2_lo, s3_lo],
                                theta=theta,
                                perform_param=self.perform_param,
                                p_list=param_list[1:],
                                e2e_enum=E2EEnum.RATE_DIFF,
                                indep=False)

    def sfa_ac_bound(self, param_list: List[float]) -> float:
        theta = param_list[0]

        foi = self.arr_list[0]
        a_2 = self.arr_list[1]
        a_3 = self.arr_list[2]

        s_1 = self.ser_list[0]
        s_2 = self.ser_list[1]
        s_3 = self.ser_list[2]

        s1_lo = LeftoverARB(ser=s_1, cross_arr=a_2)

        d_2_1 = Deconvolve(arr=a_2, ser=s_1)
        s2_lo = LeftoverARB(ser=s_2,
                            cross_arr=AggregateTwo(arr1=d_2_1, arr2=a_3))

        d_3_2 = Deconvolve(arr=a_3, ser=s_2)
        # d_3_2 = Deconvolve(arr=a_3, ser=Leftover(ser=s_2, arr=a_2))
        s3_lo = LeftoverARB(ser=s_3, cross_arr=d_3_2)

        return sfa_tandem_bound(foi=foi,
                                leftover_service_list=[s1_lo, s2_lo, s3_lo],
                                theta=theta,
                                perform_param=self.perform_param,
                                p_list=param_list[1:],
                                e2e_enum=E2EEnum.ANALYTIC_COMBINATORICS,
                                indep=False)

    def sfa_explicit(self, param_list: List[float]) -> float:
        raise NotImplementedError("This is not implemented")

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

        if server_index == 0:
            return (a_foi_rate + a_2_rate) / c_1
        elif server_index == 1:
            return (a_foi_rate + a_2_rate + a_3_rate) / c_2
        elif server_index == 2:
            return (a_foi_rate + a_3_rate) / c_3
        else:
            raise IllegalArgumentError("Wrong server index")

    def to_string(self) -> str:
        for arr in self.arr_list:
            print(arr.to_value())
        for ser in self.ser_list:
            print(ser.to_value())
        return self.to_name() + "_" + self.perform_param.__str__()


def gps_sfa_bound(param_list: [float], arr_list: List[ArrivalDistribution],
                  ser_list: List[ConstantRateServer],
                  perform_param: PerformParameter) -> float:
    theta = param_list[0]

    foi = arr_list[0]
    a_2 = arr_list[1]
    a_3 = arr_list[2]

    s_1 = ser_list[0]
    s_2 = ser_list[1]
    s_3 = ser_list[2]

    phi_list_1 = [foi.rho(theta=theta), a_2.rho(theta=theta)]
    phi_list_2 = [
        foi.rho(theta=theta),
        a_2.rho(theta=theta),
        a_3.rho(theta=theta)
    ]
    phi_list_3 = [foi.rho(theta=theta), a_3.rho(theta=theta)]

    s1_lo = LeftoverGPSPG(ser=s_1, phi_list=phi_list_1)
    s2_lo = LeftoverGPSPG(ser=s_2, phi_list=phi_list_2)
    s3_lo = LeftoverGPSPG(ser=s_3, phi_list=phi_list_3)

    conv_s1_s2 = Convolve(ser1=s1_lo, ser2=s2_lo)
    s_e2e = Convolve(ser1=conv_s1_s2, ser2=s3_lo)

    return single_hop_bound(foi=foi,
                            s_e2e=s_e2e,
                            theta=theta,
                            perform_param=perform_param)


def gps_sfa_bound_full_param(param_list: [float, float, float, float, float],
                             arr_list: List[ArrivalDistribution],
                             ser_list: List[ConstantRateServer],
                             perform_param: PerformParameter) -> float:
    if min(param_list) <= 0 or max(param_list[1:]) >= 1:
        return inf

    theta = param_list[0]

    foi = arr_list[0]

    s_1 = ser_list[0]
    s_2 = ser_list[1]
    s_3 = ser_list[2]

    phi_list_1 = [param_list[1], 1 - param_list[1]]
    phi_list_2 = [
        param_list[2], param_list[3], 1 - sum([param_list[2], param_list[3]])
    ]
    phi_list_3 = [param_list[4], 1 - param_list[4]]

    s1_lo = LeftoverGPSPG(ser=s_1, phi_list=phi_list_1)
    s2_lo = LeftoverGPSPG(ser=s_2, phi_list=phi_list_2)
    s3_lo = LeftoverGPSPG(ser=s_3, phi_list=phi_list_3)

    conv_s1_s2 = Convolve(ser1=s1_lo, ser2=s2_lo)
    s_e2e = Convolve(ser1=conv_s1_s2, ser2=s3_lo)

    return single_hop_bound(foi=foi,
                            s_e2e=s_e2e,
                            theta=theta,
                            perform_param=perform_param)


if __name__ == '__main__':
    # from nc_arrivals.qt import DM1
    # from nc_arrivals.qt import DPoisson1
    from nc_arrivals.markov_modulated import MMOODisc
    # from nc_arrivals.regulated_arrivals import LeakyBucketMassOne
    from nc_operations.perform_enum import PerformEnum
    from nc_server.constant_rate_server import ConstantRateServer
    from optimization.optimize import Optimize
    from optimization.optimize_sfa_bound import OptimizeSFABound

    from msob_and_fp.optimize_fp_bound import OptimizeFPBound
    from msob_and_fp.optimize_server_bound import OptimizeServerBound

    DELAY_PROB_TIME = PerformParameter(perform_metric=PerformEnum.DELAY,
                                       value=10**(-6))

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
        OverlappingTandem(
            arr_list=ARR_LIST,
            ser_list=SER_LIST,
            perform_param=DELAY_PROB_TIME).approximate_utilization())

    OVERLAPPING_TANDEM = OverlappingTandem(arr_list=ARR_LIST,
                                           ser_list=SER_LIST,
                                           perform_param=DELAY_PROB_TIME)

    OVERLAPPING_TANDEM_SFA = OverlappingTandemSFAPerform(
        arr_list=ARR_LIST, ser_list=SER_LIST, perform_param=DELAY_PROB_TIME)

    print("Standard Approach with PMOO:")
    print(
        Optimize(setting=OVERLAPPING_TANDEM,
                 number_param=2).grid_search(grid_bounds=[(0.1, 10.0),
                                                          (1.1, 10.0)],
                                             delta=0.1))

    print("SFA old:")
    print(
        OptimizeSFABound(setting_sfa=OVERLAPPING_TANDEM_SFA,
                         e2e_enum=E2EEnum.STANDARD,
                         number_param=3).grid_search(grid_bounds=[(0.1, 10.0),
                                                                  (1.1, 10.0),
                                                                  (1.1, 10.0)],
                                                     delta=0.1))

    print("SFA arr:")
    print(
        OptimizeSFABound(setting_sfa=OVERLAPPING_TANDEM_SFA,
                         e2e_enum=E2EEnum.ARR_RATE,
                         number_param=3).grid_search(grid_bounds=[(0.1, 10.0),
                                                                  (1.1, 10.0),
                                                                  (1.1, 10.0)],
                                                     delta=0.1))

    print("SFA min:")
    print(
        OptimizeSFABound(setting_sfa=OVERLAPPING_TANDEM_SFA,
                         e2e_enum=E2EEnum.MIN_RATE,
                         number_param=3).grid_search(grid_bounds=[(0.1, 10.0),
                                                                  (1.1, 10.0),
                                                                  (1.1, 10.0)],
                                                     delta=0.1))

    print("SFA rate diff:")
    print(
        OptimizeSFABound(setting_sfa=OVERLAPPING_TANDEM_SFA,
                         e2e_enum=E2EEnum.RATE_DIFF,
                         number_param=3).grid_search(grid_bounds=[(0.1, 10.0),
                                                                  (1.1, 10.0),
                                                                  (1.1, 17.0)],
                                                     delta=0.1))

    # print("Standard Approach with GPS:")
    # print(
    #     optimizer_perform(fun=gps_sfa_bound,
    #                       arr_list=ARR_LIST,
    #                       ser_list=SER_LIST,
    #                       perform_param=DELAY_PROB_TIME,
    #                       ranges=RANGES_1,
    #                       print_x=PRINT_X))
    #
    # print("Standard Approach with GPS fully parameterized:")
    # print(
    #     optimizer_perform(fun=gps_sfa_bound_full_param,
    #                       arr_list=ARR_LIST,
    #                       ser_list=SER_LIST,
    #                       perform_param=DELAY_PROB_TIME,
    #                       ranges=RANGES_GPS,
    #                       print_x=PRINT_X))

    print("Server Bound:")
    print(
        OptimizeServerBound(setting_msob_fp=OVERLAPPING_TANDEM,
                            number_param=1).grid_search(grid_bounds=[(0.1,
                                                                      10.0)],
                                                        delta=0.1))

    print("Flow Prolongation:")
    print(
        OptimizeFPBound(setting_msob_fp=OVERLAPPING_TANDEM,
                        number_param=1).grid_search(grid_bounds=[(0.1, 10.0)],
                                                    delta=0.1))
