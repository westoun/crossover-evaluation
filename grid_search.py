import click
from numpy import random as np_random
from quasim import Circuit, get_unitary
import random
from tqdm import tqdm
import warnings

from core.crossover import Crossover, OnePointCrossover, \
    HeadlessChickenCrossover
from core.mutation import Mutation, ReplaceGateMutation
from core.fitness import Fitness, AbsoluteDistanceFitness
from core.selection import Selection, TournamentSelection
from core.ga import GeneticAlgorithm
from core.params import ExperimentParams
from core.gate_sets import CLIFFORD_PLUS_T, CLIFFORD_PLUS_T_PLUS_I
from core.utils.random_ import random_circuit


@click.command()
@click.option(
    "--crossover",
    "-c",
    "crossover_name",
    type=click.STRING,
    default="one-point",
    help="The name of the crossover to be used. Must be either 'one-point' or 'pseudo'.",
)
@click.option(
    "--mutation_prob",
    "-mp",
    "mutation_prob",
    type=click.FLOAT,
    help="The probability of mutating a single gate.",
)
@click.option(
    "--crossover_prob",
    "-cp",
    "crossover_prob",
    type=click.FLOAT,
    help="The probability of performing crossover between two individuals.",
)
@click.option(
    "--seed",
    "-s",
    "seed",
    type=click.INT,
    default=0,
    help="The seed value to use during current experiment run.",
)
@click.option(
    "--target",
    "-target",
    "target_dir",
    type=click.STRING,
    default="results",
    help="The seed value to use during current experiment run.",
)
def run_grid_experiment(crossover_name: str, mutation_prob: float, crossover_prob: float, seed: int, target_dir: str):
    gate_count = 20
    qubit_num = 4
    population_size = 1_000
    max_generations = 1_000
    gate_set = CLIFFORD_PLUS_T_PLUS_I

    random.seed(seed)
    np_random.seed(seed)

    mutation = ReplaceGateMutation(
        qubit_num=qubit_num, gate_set=gate_set
    )

    if crossover_name == "one-point":
        crossover = OnePointCrossover()
    elif crossover_name == "pseudo":
        crossover = HeadlessChickenCrossover(
            crossover=OnePointCrossover(),
            qubit_num=qubit_num,
            gate_count=gate_count,
            gate_set=gate_set
        )
    else:
        raise NotImplementedError(
            f"No implementation found for crossover '{crossover_name}'.")

    target_circuit: Circuit = random_circuit(
        qubit_num=qubit_num, gate_count=gate_count, gate_set=gate_set
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
        gate_set=gate_set,
        seed=seed,
        result_dir=target_dir,
        tag="grid_search")

    ga = GeneticAlgorithm(params)
    ga.run()


if __name__ == "__main__":
    run_grid_experiment()
