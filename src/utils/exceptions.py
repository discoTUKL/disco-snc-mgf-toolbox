""""One file for all custom exception classes"""


class ParameterOutOfBounds(Exception):
    """Exception if input parameter is not feasible in the optimization"""
    def __init__(self, parameter):
        msg = f"Parameter is out of bounds, {parameter}"
        super(ParameterOutOfBounds, self).__init__(msg)
        self.parameter = parameter


class WrongDimension(Exception):
    """Exception if number of parameters in argument is not correct"""
    def __init__(self, parameter):
        msg = f"number of parameters {parameter} is wrong"
        super(WrongDimension, self).__init__(msg)
        self.parameter = parameter


class IllegalArgumentError(Exception):
    """Exception if number of parameters in argument is not correct"""
    def __init__(self, parameter):
        msg = f"Argument {parameter} is illegal"
        super(IllegalArgumentError, self).__init__(msg)
        self.parameter = parameter


class NotEnoughResults(Exception):
    """Exception if number of parameters in argument is not correct"""
    def __init__(self, parameter):
        msg = f"number of results {parameter} is not sufficient"
        super(NotEnoughResults, self).__init__(msg)
        self.parameter = parameter
