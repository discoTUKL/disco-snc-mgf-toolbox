""""Summarize all the performance bounds and its values in one class"""

from nc_operations.perform_enum import PerformEnum


class PerformParameter(object):
    """Performance parameter class"""
    def __init__(self, perform_metric: PerformEnum, value: float) -> None:
        self.perform_metric = perform_metric
        self.value = value

    def to_name(self) -> str:
        return str(self.perform_metric.name)

    def __str__(self) -> str:
        return f"{self.to_name()}_{str(self.value)}"

    # @property
    # def perform_metric(self):
    #     return self.perform_metric
    #
    # @perform_metric.setter
    # def perform_metric(self, new_perf_bound):
    #     self.perform_metric = new_perf_bound
    #
    # @property
    # def to_value(self):
    #     return self.to_value
    #
    # @to_value.setter
    # def to_value(self, new_value):
    #     self.to_value = new_value
