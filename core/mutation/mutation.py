
from quasim.gates import IGate


class Mutation:
    name: str

    def mutate(self, gate: IGate) -> IGate:
        raise NotImplementedError()
