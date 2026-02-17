from quasim import Circuit


class Fitness:
    name: str

    def score(self, circuit: Circuit) -> float:
        raise NotImplementedError()
