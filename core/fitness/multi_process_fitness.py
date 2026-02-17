from multiprocessing import Pool
from quasim import Circuit, get_unitary
from typing import Any, List

from .fitness import Fitness


class MultiProcessFitness(Fitness):
    name: str
    fitness: Fitness

    def __init__(self, fitness: Fitness):
        self.fitness = fitness
        self.name = f"Multi Process({fitness.name})"

    def score(self, circuits: List[Circuit]) -> List[float]:
        with Pool() as pool:
            wrapped_circuits = [
                [circuit] for circuit in circuits
            ]
            wrapped_fitness_scores = pool.map(
                self.fitness.score, wrapped_circuits)
            fitness_scores = [
                item[0] for item in wrapped_fitness_scores
            ]
            return fitness_scores
