""""One file for all custom exception classes"""


class ParameterOutOfBounds(Exception):
    """Exception if input parameter is not feasible in the optimization"""

    def __init__(self, parameter):
        msg = "Parameter is out of bounds, {0}".format(parameter)
        super(ParameterOutOfBounds, self).__init__(msg)
        self.parameter = parameter
