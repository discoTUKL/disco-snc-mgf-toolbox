"""Summarize all Simulated Annealing parameters in one class."""


class SimulAnnealParam(object):
    """Gather all simulated annealing attributes in one class."""

    def __init__(self,
                 rep_max=20,
                 temp_start=100.0,
                 cooling_factor=0.95,
                 search_radius=1.0) -> None:
        self.rep_max = rep_max
        self.temp_start = temp_start
        self.cooling_factor = cooling_factor
        self.search_radius = search_radius
