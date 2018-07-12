"""Enum class for performance metrics"""

from enum import Enum


class PerformEnum(Enum):
    """All possible performance metrics"""
    DELAY = "Delay"
    DELAY_PROB = "DelayProb"
    OUTPUT = "Output"
    BACKLOG = "Backlog"
    BACKLOG_PROB = "BacklogProb"
