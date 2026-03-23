
from copy import deepcopy
from quasim import Circuit
from random import randint, sample
from typing import Tuple

from .crossover import Crossover


class TwoPointCrossover(Crossover):
    name: str = "Two Point Crossover"

    def cross(self, circuit1: Circuit, circuit2: Circuit) -> Tuple[Circuit, Circuit]:

        max_position = min(len(circuit1.gates), len(circuit2.gates))

        crx_i1, crx_i2 = sample(range(max_position), k=2)
        if crx_i1 > crx_i2:
            crx_i1, crx_i2 = crx_i2, crx_i1

        circuit1.gates[crx_i1:crx_i2], circuit2.gates[crx_i1:crx_i2] = (
            deepcopy(circuit2.gates[crx_i1:crx_i2]),
            deepcopy(circuit1.gates[crx_i1:crx_i2]),
        )

        return circuit1, circuit2
