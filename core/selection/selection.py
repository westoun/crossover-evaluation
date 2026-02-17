
from quasim import Circuit
from typing import List


class Selection:
    name: str

    def select(self, circuits: List[Circuit], fitness_scores: List[float], k: int) -> List[Circuit]:
        raise NotImplementedError()
