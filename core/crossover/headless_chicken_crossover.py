
from copy import deepcopy
from quasim import Circuit
from quasim.gates import IGate
from random import randint, random
from typing import Tuple, List

from .crossover import Crossover
from core.utils.random_ import random_circuit


class HeadlessChickenCrossover(Crossover):
    name: str

    crossover: Crossover
    qubit_num: int
    gate_count: int
    gate_set: List[IGate]

    def __init__(self, crossover: Crossover, qubit_num: int, gate_count: int, gate_set: List[IGate]):
        self.crossover = crossover
        self.qubit_num = qubit_num
        self.gate_count = gate_count
        self.gate_set = gate_set

        self.name = f"Headless Chicken({crossover.name})"

    def cross(self, circuit1: Circuit, circuit2: Circuit) -> Tuple[Circuit, Circuit]:
        children = []

        for circuit in [circuit1, circuit2]:
            pseudo_parent = random_circuit(
                qubit_num=self.qubit_num, gate_count=self.gate_count, gate_set=self.gate_set
            )

            pseudo_child1, pseudo_child2 = self.crossover.cross(
                circuit, pseudo_parent
            )

            if random() < 0.5:
                children.append(pseudo_child1)
            else:
                children.append(pseudo_child2)

        return children
