from enum import Enum


class OptMethod(Enum):
    GRID_SEARCH = "GridSearch"
    NELDER_MEAD = "NelderMead"
    PATTERN_SEARCH = "PatternSearch"
    BASIN_HOPPING = "BasinHopping"
    SIMULATED_ANNEALING = "SimulatedAnnealing"
    BFGS = "BFGS"
    GS_OLD = "GridSearchOld"
    NM_OLD = "NelderMeadOld"
