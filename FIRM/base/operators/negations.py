from enum import Enum
import numpy as np


## Store all the important triangular norms

class NegationsExamples(Enum):
    CLASSICAL = "classical_tnorm"
    SUGENO = "sugeno_tnorm"

    @staticmethod
    def get_negation(negation):
        if negation == NegationsExamples.CLASSICAL:
            return NegationsExamples.classical_negation
        if negation == NegationsExamples.SUGENO:
            return NegationsExamples.sugeno_negation

    @staticmethod
    def classical_negation(x: float) -> float:
        return 1-x

    @staticmethod
    def sugeno_negation(x: float, w: float) -> float:
        return (1-x)/(1-w*x)