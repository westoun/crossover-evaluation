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

    mutation_prob = 0.01
    crossover_prob = 0.3
    seed = 0

    random.seed(seed)
    np_random.seed(seed)

    mutation = ReplaceGateMutation(
        qubit_num=qubit_num, gate_set=CLIFFORD_PLUS_T
    )

    # crossover = HeadlessChickenCrossover(
    #     crossover=OnePointCrossover(),
    #     qubit_num=qubit_num,
    #     gate_count=gate_count,
    #     gate_set=CLIFFORD_PLUS_T
    # )
    crossover = OnePointCrossover()

    target_circuit: Circuit = random_circuit(
        qubit_num=qubit_num, gate_count=gate_count, gate_set=CLIFFORD_PLUS_T
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target_unitary = get_unitary(target_circuit)

    # fitness = MultiProcessFitness(
    #     fitness=AbsoluteDistanceFitness(
    #         target_unitary=target_unitary
    #     )
    # )
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
        tag="2")

    ga = GeneticAlgorithm(params)
    ga.run()
