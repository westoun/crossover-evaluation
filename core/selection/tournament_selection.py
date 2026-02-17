from copy import deepcopy
from quasim import Circuit
from random import sample
from typing import List

from .selection import Selection


class TournamentSelection(Selection):
    name: str = "Tournament Selection"

    tournament_size: int

    def __init__(self, tournament_size: int = 2):
        self.tournament_size = tournament_size

    def select(self, circuits: List[Circuit], fitness_scores: List[float], k: int) -> List[Circuit]:
        assert len(circuits) == len(fitness_scores)

        selection = []

        for _ in range(k):
            candidate_indices: List[int] = sample(
                population=range(len(circuits)), k=self.tournament_size)
            scores = [fitness_scores[i] for i in candidate_indices]

            winner_score = min(scores)
            winner_idx = fitness_scores.index(winner_score)

            winner = circuits[winner_idx]
            selection.append(deepcopy(winner))

        return selection
