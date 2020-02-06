"""Class for all arrival processes that cannot be described via (sigma, rho)"""


def expect_dm1(delta_time: int, lamb: float) -> float:
    return delta_time / lamb


def var_dm1(delta_time: int, lamb: float) -> float:
    return delta_time / (lamb**2)
