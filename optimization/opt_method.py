from enum import Enum


class OptMethod(Enum):
    GRID_SEARCH = "GridSearch"
    NELDER_MEAD = "NelderMead"
    PATTERN_SEARCH = "PatternSearch"
    SIMULATED_ANNEALING = "SimulatedAnnealing"
    BFGS = "BFGS"
