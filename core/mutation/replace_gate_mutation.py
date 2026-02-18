from quasim import Circuit
from quasim.gates import IGate
from random import randint
from typing import List

from .mutation import Mutation
from core.utils.random_ import random_gate


class ReplaceGateMutation(Mutation):
    name: str = "Replace Gate Mutation"

    qubit_num: int
    gate_set: List[IGate]

    def __init__(self, qubit_num: int, gate_set: List[IGate]):
        self.qubit_num = qubit_num
        self.gate_set = gate_set

    def mutate(self, gate: IGate) -> IGate:
        return random_gate(
            self.qubit_num, self.gate_set)
