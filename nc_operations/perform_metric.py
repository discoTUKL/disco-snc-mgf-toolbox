"""Enum class for performance metrics"""

from enum import Enum


class PerformMetric(Enum):
    """All possible performance metrics as enums"""
    DELAY = "Delay"
    DELAY_PROB = "DelayProb"
    OUTPUT = "Output"
    BACKLOG = "Backlog"
    BACKLOG_PROB = "BacklogProb"
