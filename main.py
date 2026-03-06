from numpy import random as np_random
from quasim import Circuit, get_unitary
import random
import warnings

from core.crossover import Crossover, OnePointCrossover, \
    HeadlessChickenCrossover
from core.mutation import Mutation, ReplaceGateMutation
from core.fitness import Fitness, AbsoluteDistanceFitness, \
    MultiProcessFitness
from core.selection import Selection, TournamentSelection
from core.ga import GeneticAlgorithm
from core.params import ExperimentParams
from core.gate_sets import CLIFFORD_PLUS_T
from core.utils.random_ import random_circuit


if __name__ == "__main__":
    gate_count = 10
    qubit_num = 3
    population_size = 1000
    max_generations = 1000

    seed_num = 4
    seed_offset = 1

    mutation = ReplaceGateMutation(
        qubit_num=qubit_num, gate_set=CLIFFORD_PLUS_T
    )

    hc_crossover = HeadlessChickenCrossover(
        crossover=OnePointCrossover(),
        qubit_num=qubit_num,
        gate_count=gate_count,
        gate_set=CLIFFORD_PLUS_T
    )
    op_crossover = OnePointCrossover()

    for seed in range(seed_num):
        seed = seed + seed_offset

        random.seed(seed)
        np_random.seed(seed)

        for mutation_prob, crossover_prob, crossover in [
            (0.05, 0.3, op_crossover),
            (0.05, 0.3, hc_crossover),
            (1.0, 0.0, op_crossover),
        ]:

            target_circuit: Circuit = random_circuit(
                qubit_num=qubit_num, gate_count=gate_count, gate_set=CLIFFORD_PLUS_T
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                target_unitary = get_unitary(target_circuit)

            fitness = AbsoluteDistanceFitness(
                target_unitary=target_unitary
            )

            selection = TournamentSelection(tournament_size=2)

            params = ExperimentParams(
                crossover=crossover,
                mutation=mutation,
                mutation_prob=mutation_prob,
                crossover_prob=crossover_prob,
                fitness=fitness,
                selection=selection,
                population_size=population_size,
                max_generations=max_generations,
                qubit_num=qubit_num,
                gate_count=gate_count,
                gate_set=CLIFFORD_PLUS_T,
                seed=seed,
                result_dir="results",
                tag=None)

            ga = GeneticAlgorithm(params)
            ga.run()
