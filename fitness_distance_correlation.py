
from copy import deepcopy
import matplotlib.pyplot as plt
from numpy import random as np_random
from quasim import Circuit, get_unitary
from quasim.gates import IGate
import random
from statistics import mean, stdev
from scipy import stats
from tqdm import tqdm
from typing import List
import warnings

from core.utils.random_ import random_circuit, random_gate
from core.utils.logging import save_to_json, get_timestamp
from core.fitness import Fitness, AbsoluteDistanceFitness
from core.gate_sets import CLIFFORD_PLUS_T
from core.mutation import ReplaceGateMutation


def get_gate_type(gate: IGate) -> str:
    return str(type(gate)).split(".")[-1].split("'")[0]


if __name__ == "__main__":
    seed_num = 100
    qubit_num = 4
    gate_count = 20

    max_distance = 5
    circuits_per_distance = 100

    mutation = ReplaceGateMutation(
        qubit_num=qubit_num, gate_set=CLIFFORD_PLUS_T
    )

    distances = []
    fitness_scores = []

    for seed in tqdm(range(seed_num), total=seed_num):
        random.seed(seed)
        np_random.seed(seed)

        target_circuit: Circuit = random_circuit(
            qubit_num=qubit_num, gate_count=gate_count, gate_set=CLIFFORD_PLUS_T
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target_unitary = get_unitary(target_circuit)

        fitness = AbsoluteDistanceFitness(
            target_unitary=target_unitary
        )

        current_bin = [target_circuit]

        for distance in range(1, max_distance + 1):
            parents = random.choices(
                population=current_bin, k=circuits_per_distance)

            children = []
            for circuit in parents:
                child = deepcopy(circuit)

                gate_i = random.randint(0, len(child.gates) - 1)
                child.gates[gate_i] = random_gate(qubit_num, CLIFFORD_PLUS_T)
                children.append(child)

            child_fitnesses = fitness.score(children)

            distances.extend([
                distance for _ in children
            ])
            fitness_scores.extend(child_fitnesses)

            current_bin = children

    correlation = stats.pearsonr(
        fitness_scores, distances).correlation
    print(f"Fitness distance correlation: {correlation}")

    fig, ax = plt.subplots()

    # sort by dict

    fitness_scores_per_distance = []
    for _ in list(range(max_distance)):
        fitness_scores_per_distance.append([])

    for distance, fitness_score in zip(distances, fitness_scores):
        fitness_scores_per_distance[distance - 1].append(fitness_score)

    ax.boxplot(fitness_scores_per_distance,
               tick_labels=list(range(1, max_distance + 1)))

    plt.xlabel("distance")
    plt.ylabel("fitness score")

    plt.ylim(0)

    plt.grid()

    plt.savefig("results/fdc.png", bbox_inches='tight')
    plt.clf()
