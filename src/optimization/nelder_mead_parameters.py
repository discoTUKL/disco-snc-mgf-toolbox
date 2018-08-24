"""Summarize all Nelder-Mead parameters in one class"""


class NelderMeadParameters(object):
    def __init__(self,
                 reflection_alpha=1.0,
                 expansion_gamma=2.0,
                 contraction_beta=0.5,
                 shrink_gamma=0.5) -> None:
        self.reflection_alpha = reflection_alpha
        self.expansion_gamma = expansion_gamma
        self.contraction_beta = contraction_beta
        self.shrink_gamma = shrink_gamma
