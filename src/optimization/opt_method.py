from enum import Enum


class OptMethod(Enum):
    GRID_SEARCH = "GridSearch"
    PATTERN_SEARCH = "PatternSearch"
    NELDER_MEAD = "NelderMead"
    BASIN_HOPPING = "BasinHopping"
    DIFFERENTIAL_EVOLUTION = "DifferentialEvolution"
    DUAL_ANNEALING = "DualAnnealing"
    SIMULATED_ANNEALING = "SimulatedAnnealing"
    GS_OLD = "GridSearchOld"
    NM_OLD = "NelderMeadOld"
    BFGS = "BFGS"
