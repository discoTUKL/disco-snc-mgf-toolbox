"""The perform_parameter counterpart with lists"""

from nc_operations.perform_metric import PerformMetric


class PerformParamList(object):
    """"Performance parameter list class"""

    def __init__(self, perform_metric: PerformMetric, values_list) -> None:
        # Don't use type list as it does not work with 'range'
        self.perform_metric = perform_metric
        self.values_list = values_list
