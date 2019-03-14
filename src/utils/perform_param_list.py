"""The perform_parameter counterpart with lists"""

from nc_operations.perform_enum import PerformEnum
from utils.perform_parameter import PerformParameter


class PerformParamList(object):
    """"Performance parameter list class"""

    def __init__(self, perform_metric: PerformEnum, values_list) -> None:
        # IMPORTANT: Don't use type "list" as it does not work with "range"
        self.perform_metric = perform_metric
        self.values_list = values_list

    def to_name(self) -> str:
        return str(self.perform_metric.name)

    def get_parameter_at_i(self, i: int) -> PerformParameter:
        return PerformParameter(
            perform_metric=self.perform_metric, value=self.values_list[i])

    def __len__(self) -> int:
        return len(self.values_list)

    def __str__(self) -> str:
        return str(self.perform_metric.name) + "_LIST"
