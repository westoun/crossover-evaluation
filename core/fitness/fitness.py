from quasim import Circuit


class Fitness:

    def score(self, circuit: Circuit) -> float:
        raise NotImplementedError()
