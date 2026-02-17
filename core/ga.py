
from copy import deepcopy
from statistics import mean, stdev
import random

from .utils.random_ import random_circuit
from .params import ExperimentParams
from .utils.logging import log_params, log_fitness


class GeneticAlgorithm:
    params: ExperimentParams

    def __init__(self, params: ExperimentParams):
        self.params = params

    def run(self):
        log_params(self.params)

        population = [
            random_circuit(
                qubit_num=self.params.qubit_num,
                gate_count=self.params.gate_count,
                gate_set=self.params.gate_set
            )
            for _ in range(self.params.population_size)
        ]

        for generation in range(1, self.params.max_generations + 1):
            offspring = [deepcopy(circuit) for circuit in population]

            if generation > 1:
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

            best_fitness = min(fitness_scores)
            mean_fitness = mean(fitness_scores)
            fitness_stdev = stdev(fitness_scores)
            log_fitness(
                generation=generation,
                best_fitness=best_fitness,
                mean_fitness=mean_fitness,
                fitness_stdev=fitness_stdev,
                params=self.params
            )

            population = self.params.selection.select(
                offspring, fitness_scores, k=self.params.population_size
            )
