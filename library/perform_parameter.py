""""Summarize all the performance bounds and its values in one class"""

from nc_operations.perform_metric import PerformMetric


class PerformParameter(object):
    """Performance parameter class"""

    def __init__(self, perform_metric: PerformMetric, value) -> None:
        self.perform_metric = perform_metric
        self.value = value

    def to_string(self) -> str:
        return str(self.perform_metric) + "_" + str(self.value)

    # TODO: get to know how to use @property
    # @property
    # def perform_metric(self):
    #     return self.perform_metric
    #
    # @perform_metric.setter
    # def perform_metric(self, new_perf_bound):
    #     self.perform_metric = new_perf_bound
    #
    # @property
    # def value(self):
    #     return self.value
    #
    # @value.setter
    # def value(self, new_value):
    #     self.value = new_value
