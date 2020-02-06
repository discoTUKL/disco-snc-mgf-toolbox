"""Class for all service processes that cannot be described via (sigma, rho)"""


def expect_const_rate(delta_time: int, rate: float) -> float:
    return delta_time * rate
