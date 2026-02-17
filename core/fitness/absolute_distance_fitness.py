from quasim import Circuit, get_unitary
import warnings
from typing import Any, List

from .fitness import Fitness


class AbsoluteDistanceFitness(Fitness):
    name: str = "Absolute Distance Fitness"

    target_unitary: Any

    def __init__(self, target_unitary: Any):
        self.target_unitary = target_unitary

    def score(self, circuits: List[Circuit]) -> List[float]:
        fitness_scores = []

        for circuit in circuits:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate_unitary = get_unitary(circuit)

            (c_rows, c_cols) = candidate_unitary.shape
            (t_rows, t_cols) = self.target_unitary.shape

            assert c_rows == t_rows
            assert c_cols == t_cols

            distance = 0
            for row_i in range(c_rows):
                for col_i in range(c_cols):
                    distance += abs(candidate_unitary[row_i]
                                    [col_i] - self.target_unitary[row_i][col_i])

            fitness_scores.append(distance)

        return fitness_scores
