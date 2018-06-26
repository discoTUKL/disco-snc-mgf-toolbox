"""Compare with alternative traffic description"""

from library.perform_parameter import PerformParameter
from nc_operations.perform_metric import PerformMetric
from nc_operations.performance_bounds_alternative import del_prob_alter_opt
from nc_processes.regulated_arrivals import (LeakyBucketMassOne,
                                             TokenBucketConstant)
from nc_processes.service_distribution import ConstantRate
from optimization.optimize import Optimize
from single_server.single_server_perform import SingleServerPerform

if __name__ == '__main__':
    DELAYPROB6 = PerformParameter(
        perform_metric=PerformMetric.DELAY_PROB, value=5)

    NUMBER_AGGREGATIONS = 5

    RHO_SINGLE = 0.1
    SERVICE_RATE = 2.5
    SIGMA_SINGLE = 10.0

    bound_list = [(0.05, 15.0)]
    delta = 0.05
    print_x = True

    constant_rate_server = ConstantRate(SERVICE_RATE)

    tb_const = TokenBucketConstant(
        sigma_single=SIGMA_SINGLE,
        rho_single=RHO_SINGLE,
        n=NUMBER_AGGREGATIONS)

    const_single = SingleServerPerform(
        arr=tb_const, ser=constant_rate_server, perform_param=DELAYPROB6)

    leaky_mass_1 = SingleServerPerform(
        arr=LeakyBucketMassOne(
            sigma_single=SIGMA_SINGLE,
            rho_single=RHO_SINGLE,
            n=NUMBER_AGGREGATIONS),
        ser=constant_rate_server,
        perform_param=DELAYPROB6)

    const_opt = Optimize(
        setting=const_single, print_x=print_x).grid_search(
            bound_list=bound_list, delta=delta)
    print("const_opt", const_opt)

    leaky_mass_1_opt = Optimize(
        setting=leaky_mass_1, print_x=print_x).grid_search(
            bound_list=bound_list, delta=delta)
    print("leaky_mass_1_opt", leaky_mass_1_opt)

    leaky_bucket_alter_opt = del_prob_alter_opt(
        delay_value=5, t=14, print_x=print_x)
    print("leaky_bucket_alter_opt", leaky_bucket_alter_opt)
