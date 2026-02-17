
from dataclasses import dataclass
from copy import deepcopy
from quasim.gates import IGate
import random
from typing import List

from .mutation import Mutation, ReplaceGateMutation
from .crossover import Crossover
from .fitness import Fitness
from .selection import Selection
from .utils.random_ import generate_random_circuit
from .gate_sets import CLIFFORD_PLUS_T


@dataclass
class GAParams:
    crossover: Crossover
    mutation: Mutation

    mutation_prob: float
    crossover_prob: float

    fitness: Fitness
    selection: Selection

    population_size: int
    max_generations: int

    qubit_num: int
    gate_count: int
    gate_set: List[IGate]


class GeneticAlgorithm:
    params: GAParams

    def __init__(self, params: GAParams):
        self.params = params

    def run(self):
        population = [
            generate_random_circuit(
                qubit_num=self.params.qubit_num,
                gate_count=self.params.gate_count,
                gate_set=self.params.gate_set
            )
            for _ in range(self.params.population_size)
        ]

        for generation in range(1, self.params.max_generations + 1):
            offspring = [deepcopy(circuit) for circuit in population]

            # Shuffle to avoid crossover in the same proximity across
            # generations.
            random.shuffle(offspring)

            for i in range(0, len(offspring), 2):
                if random.random() < self.params.crossover_prob:
                    offspring[i], offspring[i +
                                            1] = self.params.crossover.cross(offspring[i], offspring[i + 1])

            for i, circuit in enumerate(offspring):
                if random.random() < self.params.mutation_prob:
                    offspring[i] = self.params.mutation.mutate(circuit)

            fitness_scores = [
                self.params.fitness.score(circuit) for circuit in offspring
            ]

            population = self.params.selection.select(
                offspring, fitness_scores, k=self.params.population_size
            )
