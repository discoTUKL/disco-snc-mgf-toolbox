"""Enum of Change / Improvement Metrics"""

from enum import Enum


class ChangeEnum(Enum):
    RATIO_NEW_REF = "Ratio of New Approach vs. Reference"
    RATIO_REF_NEW = "Ratio of Reference vs New Approach"
    RELATIVE_CHANGE = "Ratio of Change vs. Reference"
    DIFF_REF_NEW = "Difference of Reference vs. New Approach"
