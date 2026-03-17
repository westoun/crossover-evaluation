import click
import math
import numpy as np
from numpy import random as np_random
from quasim import Circuit, get_unitary
import random
import warnings

from core.utils.random_ import random_circuit
from core.gate_sets import CLIFFORD_PLUS_T, CLIFFORD_PLUS_T_PLUS_I
from core.params import ExperimentParams
from core.ga import GeneticAlgorithm
from core.selection import Selection, TournamentSelection
from core.fitness import Fitness, AbsoluteDistanceFitness, \
    AbsoluteStateDistanceFitness
from core.mutation import Mutation, ReplaceGateMutation
from core.crossover import Crossover, OnePointCrossover, \
    HeadlessChickenCrossover


def create_qft_unitary(qubit_num: int) -> np.ndarray:
    dim = 2 ** qubit_num

    dft_matrix = np.zeros((dim, dim), dtype=np.complex128)

    w = np.pow(np.e, 2 * np.pi * 1j / dim)

    for i in range(dim):
        for j in range(dim):
            dft_matrix[i, j] = np.pow(w, i * j)

    unitary = 1 / np.pow(dim, 0.5) * dft_matrix
    return unitary


def create_haar_random_unitary(qubit_num: int) -> np.ndarray:
    dim = 2 ** qubit_num

    Z = (np.random.normal(size=(dim, dim)) + 1j *
         np.random.normal(size=(dim, dim))) / math.sqrt(2)
    Q, R = np.linalg.qr(Z)
    D = np.diag([
        R[i, i] / np.abs(R[i, i]) for i in range(dim)
    ])
    return np.dot(Q, D)


def create_haar_random_state(qubit_num: int) -> np.ndarray:
    U = create_haar_random_unitary(qubit_num)
    return U[0]


def create_w_state(qubit_num: int) -> np.ndarray:
    dim = 2 ** qubit_num

    state = np.zeros(dim, dtype=np.complex128)

    for i in range(qubit_num):
        state[2**i] = 1

    state = state / math.sqrt(qubit_num)
    return state


def create_ghz_state(qubit_num: int) -> np.ndarray:
    dim = 2 ** qubit_num

    state = np.zeros(dim, dtype=np.complex128)

    state[0] = 1
    state[dim - 1] = 1

    state = state / math.sqrt(2)
    return state


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
    "--gate_count",
    "-gc",
    "gate_count",
    type=click.INT,
    default=20,
    help="The number of gates per circuit.",
)
@click.option(
    "--qubit_num",
    "-qn",
    "qubit_num",
    type=click.INT,
    default=4,
    help="The number of qubits per circuit.",
)
@click.option(
    "--population",
    "-p",
    "population_size",
    type=click.INT,
    default=1_000,
    help="The number of circuits in the population.",
)
@click.option(
    "--generations",
    "-g",
    "max_generations",
    type=click.INT,
    default=1_000_000,
    help="The number of generations for which the ga is to be run.",
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
    "--result-dir",
    "-rd",
    "result_dir",
    type=click.STRING,
    default="results",
    help="The dir path where result files are to be stored.",
)
@click.option(
    "--target",
    "-t",
    "target",
    type=click.STRING,
    default="qft",
    help="The synthesis target. Must be either 'random', 'qft', 'haar', 'w-state', 'ghz-state', or 'haar-state'.",
)
@click.option(
    "--tag",
    "-tag",
    "tag",
    type=click.STRING,
    default=None,
    help="An optional tag to later identify experiments by.",
)
def run_experiment(crossover_name: str, mutation_prob: float, crossover_prob: float, gate_count: int,
                   qubit_num: int, population_size: int, max_generations: int, seed: int, result_dir: str,
                   target: str, tag: str):
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

    if target == "qft":
        target_unitary = create_qft_unitary(qubit_num)
        fitness = AbsoluteDistanceFitness(
            target_unitary=target_unitary
        )

    elif target == "random":
        target_circuit: Circuit = random_circuit(
            qubit_num=qubit_num, gate_count=gate_count, gate_set=gate_set
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target_unitary = get_unitary(target_circuit)

        fitness = AbsoluteDistanceFitness(
            target_unitary=target_unitary
        )

    elif target == "haar":
        target_unitary = create_haar_random_unitary(qubit_num)

        fitness = AbsoluteDistanceFitness(
            target_unitary=target_unitary
        )

    elif target == "haar-state":
        target_state = create_haar_random_state(qubit_num)
        fitness = AbsoluteStateDistanceFitness(target_state)

    elif target == "ghz-state":
        target_state = create_ghz_state(qubit_num)

        fitness = AbsoluteStateDistanceFitness(target_state)

    elif target == "w-state":
        target_state = create_w_state(qubit_num)

        fitness = AbsoluteStateDistanceFitness(target_state)

    else:
        raise NotImplementedError(
            f"No implementation found for target '{target}'.")

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
        result_dir=result_dir,
        tag=tag)

    ga = GeneticAlgorithm(params)
    ga.run()


if __name__ == "__main__":
    run_experiment()
