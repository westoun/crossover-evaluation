from quasim import Circuit
from typing import List


class Fitness:
    name: str

    def score(self, circuits: List[Circuit]) -> List[float]:
        raise NotImplementedError()
