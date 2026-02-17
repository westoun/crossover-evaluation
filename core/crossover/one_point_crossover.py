
from copy import deepcopy
from quasim import Circuit
from random import randint
from typing import Tuple

from .crossover import Crossover


class OnePointCrossover(Crossover):
    name: str = "One Point Crossover"

    def cross(cls, circuit1: Circuit, circuit2: Circuit) -> Tuple[Circuit, Circuit]:

        max_position = min(len(circuit1.gates), len(circuit2.gates))
        crx_idx = randint(1, max_position - 1)

        circuit1.gates[crx_idx:], circuit2.gates[crx_idx:] = (
            deepcopy(circuit2.gates[crx_idx:]),
            deepcopy(circuit1.gates[crx_idx:]),
        )

        return circuit1, circuit2
