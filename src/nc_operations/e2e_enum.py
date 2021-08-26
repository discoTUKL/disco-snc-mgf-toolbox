"""Enum class for all sfa bounds"""

from enum import Enum


class E2EEnum(Enum):
    """Different end-to-end bounds"""
    STANDARD = "Standard"
    ARR_RATE = "ArrivalRate"
    MIN_RATE = "MinRate"
    RATE_DIFF = "RateDifference"
    ANALYTIC_COMBINATORICS = "AnalyticCombinatorics"
    CUTTING = "CuttingTechnique"
