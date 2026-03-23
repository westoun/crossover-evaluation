
from copy import deepcopy
from quasim import Circuit
from random import randint
from typing import Tuple

from .crossover import Crossover


class UniformCrossover(Crossover):
    name: str = "Uniform Crossover"

    def cross(self, circuit1: Circuit, circuit2: Circuit) -> Tuple[Circuit, Circuit]:

        max_position = min(len(circuit1.gates), len(circuit2.gates))

        for gate_i in range(max_position):
            circuit1.gates[gate_i], circuit2.gates[gate_i] = deepcopy(
                circuit2.gates[gate_i]), deepcopy(circuit1.gates[gate_i])

        return circuit1, circuit2
