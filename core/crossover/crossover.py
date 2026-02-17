
from quasim import Circuit
from typing import Tuple


class Crossover:
    name: str

    def cross(self, circuit1: Circuit, circuit2: Circuit) -> Tuple[Circuit, Circuit]:
        raise NotImplementedError()
