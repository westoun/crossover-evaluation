import click
from numpy import random as np_random
from quasim import Circuit, get_unitary
import random
import warnings

from core.crossover import Crossover, OnePointCrossover, \
    HeadlessChickenCrossover
from core.mutation import Mutation, ReplaceGateMutation
from core.fitness import Fitness, AbsoluteDistanceFitness
from core.selection import Selection, TournamentSelection
from core.ga import GeneticAlgorithm
from core.params import ExperimentParams
from core.gate_sets import CLIFFORD_PLUS_T
from core.utils.random_ import random_circuit

# @click.command()
# @click.option(
#     "--qubits",
#     "-q",
#     "qubit_num",
#     type=click.INT,
#     help="The amount of qubits per circuit.",
# )
# @click.option(
#     "--gates",
#     "-g",
#     "gate_count",
#     type=click.INT,
#     help="The amount of gates per circuit.",
# )
# @click.option(
#     "--crossover",
#     "-c",
#     "crossover",
#     type=click.INT,
#     help="The amount of gates per circuit.",
# )
# @click.option(
#     "--seed",
#     "-s",
#     "seed",
#     type=click.INT,
#     default=0,
#     help="The seed value to use.",
# )
# def run_experiment(qubit_num: int, gate_count: int, crossover: str, seed: int = 0) -> None:
#     population_size: int = 1000
#     max_generations: int = 100_000

#     crossover_prob: float = 0.3
#     mutation_prob: float = 0.05


if __name__ == "__main__":
    seed = 0

    random.seed(seed)
    np_random.seed(seed)

    gate_count = 10
    qubit_num = 4
    population_size = 100

    mutation = ReplaceGateMutation(
        qubit_num=qubit_num, gate_set=CLIFFORD_PLUS_T
    )

    crossover = HeadlessChickenCrossover(
        crossover=OnePointCrossover(),
        qubit_num=qubit_num,
        gate_count=gate_count,
        gate_set=CLIFFORD_PLUS_T
    )
    # crossover = OnePointCrossover()

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
        mutation_prob=0.05,
        crossover_prob=0.4,
        fitness=fitness,
        selection=selection,
        population_size=population_size,
        max_generations=1000,
        qubit_num=qubit_num,
        gate_count=gate_count,
        gate_set=CLIFFORD_PLUS_T,
        seed=0,
        result_dir="results",
        tag="small_mutprob")

    ga = GeneticAlgorithm(params)
    ga.run()
