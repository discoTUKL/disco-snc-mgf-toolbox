"""Enum of Distributions for Monte Carlo Parameter Evaluation"""

from enum import Enum


class MCEnum(Enum):
    UNIFORM = "Uniform"
    EXPONENTIAL = "Exponential"
    PARETO = "Pareto"
    LOG_NORMAL = "Log-Normal"
    CHI_SQUARED = "Chi-squared"
