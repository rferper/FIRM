from enum import Enum
import numpy as np


## Store all the considered triangular norms

class TnormsExamples(Enum):
    MINIMUM = "minimum_tnorm"
    PRODUCT = "product_tnorm"
    SCHWEIZER_SKLAR = "Schweizer_Sklar_tnorm"
    HAMACHER = "hamacher_tnorm"
    FRANK = "frank_tnorm"
    LUKASIEWICZ = "lukasiewicz_tnorm"

    @staticmethod
    def get_tnorm(tnorm):
        if tnorm == TnormsExamples.MINIMUM:
            return TnormsExamples.minimum_tnorm
        if tnorm == TnormsExamples.PRODUCT:
            return TnormsExamples.product_tnorm
        if tnorm == TnormsExamples.SCHWEIZER_SKLAR:
            return TnormsExamples.ss_tnorm
        if tnorm == TnormsExamples.HAMACHER:
            return TnormsExamples.h_tnorm
        if tnorm == TnormsExamples.FRANK:
            return TnormsExamples.f_tnorm
        if tnorm == TnormsExamples.LUKASIEWICZ:
            return TnormsExamples.luka_tnorm

    @staticmethod
    def minimum_tnorm(x: float, y: float) -> float:
        return np.minimum(x, y)

    @staticmethod
    def product_tnorm(x: float, y: float) -> float:
        return x * y

    @staticmethod
    def ss_tnorm(x: float, y: float, mu: float) -> float:
        if x == 0 or y == 0:
            return 0
        else:
            return (np.maximum(x ** mu + y ** mu - 1, 0)) ** (1 / mu)

    @staticmethod
    def h_tnorm(x: float, y: float, mu: float) -> float:
        return (x * y) / (mu + (1 - mu) * (x + y - x * y))

    @staticmethod
    def f_tnorm(x: float, y: float, mu: float) -> float:
        if x == 0 or y == 0:
            return 0
        else:
            return np.log(1 + (mu ** x - 1) * (mu ** y - 1) / (mu - 1)) / np.log(mu)

    @staticmethod
    def luka_tnorm(x: float, y: float) -> float:
        if x == 0 or y == 0:
            return 0
        else:
            return np.maximum(0, x + y - 1)
