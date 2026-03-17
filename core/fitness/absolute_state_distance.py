from quasim import Circuit, get_unitary
import warnings
from typing import Any, List

from .fitness import Fitness
from .cache import Cache


class AbsoluteStateDistanceFitness(Fitness):
    name: str = "Absolute State Distance Fitness"

    target_state: Any
    cache: Cache

    def __init__(self, target_state: Any):
        self.target_state = target_state
        self.cache = Cache()

    def score(self, circuits: List[Circuit]) -> List[float]:
        fitness_scores = []

        for circuit in circuits:

            if circuit in self.cache:
                fitness_scores.append(self.cache.get(circuit))
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate_unitary = get_unitary(circuit)

            (c_rows, c_cols) = candidate_unitary.shape

            assert c_cols == len(self.target_state)

            distance = 0
            for col_i in range(c_cols):
                distance += abs(candidate_unitary[0]
                                [col_i] - self.target_state[col_i])

            fitness_scores.append(distance)
            self.cache.add(circuit, distance)

        return fitness_scores
