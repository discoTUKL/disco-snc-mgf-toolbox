""""Summarize all the performance bounds and its values in one class"""

from nc_operations.perform_enum import PerformEnum


class PerformParameter(object):
    """Performance parameter class"""

    def __init__(self, perform_metric: PerformEnum, value: float) -> None:
        self.perform_metric = perform_metric
        self.value = value

    def to_name(self) -> str:
        return str(self.perform_metric.name)

    def to_name_value(self) -> str:
        return f"{self.to_name()}_{str(self.value)}"
