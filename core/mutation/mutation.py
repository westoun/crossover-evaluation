
from quasim import Circuit
from quasim.gates import IGate
from typing import List


class Mutation:
    name: str

    def mutate(cls, circuit: Circuit) -> Circuit:
        raise NotImplementedError()
