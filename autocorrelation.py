from copy import deepcopy
import math
import matplotlib.pyplot as plt
from numpy import random as np_random
from quasim import Circuit, get_unitary
from quasim.gates import IGate
import random
from scipy import stats
from tqdm import tqdm
from typing import List
import warnings

from core.utils.random_ import random_circuit, random_gate
from core.fitness import Fitness, AbsoluteDistanceFitness
from core.gate_sets import CLIFFORD_PLUS_T


def get_gate_type(gate: IGate) -> str:
    return str(type(gate)).split(".")[-1].split("'")[0]


def get_differing_gate(old_circuit: Circuit, new_circuit: Circuit) -> IGate:

    assert len(old_circuit.gates) == len(new_circuit.gates)

    for old_gate, new_gate in zip(old_circuit.gates, new_circuit.gates):
        if old_gate.__repr__() != new_gate.__repr__():
            return new_gate


def compute_correlation_length(autocorrelation: float) -> float:
    correlation_length = -1 / math.log(autocorrelation)
    return correlation_length


if __name__ == "__main__":

    seed_num = 1000
    qubit_num = 4
    gate_count = 20

    walk_length = 100
    max_offset = 10

    all_gate_types: List[List] = []
    all_fitness_scores: List[List] = []

    # Create random walk data
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

        start_circuit: Circuit = random_circuit(
            qubit_num=qubit_num, gate_count=gate_count, gate_set=CLIFFORD_PLUS_T
        )

        random_walk = [start_circuit]
        random_walk_gates = [None]
        for _ in range(walk_length):
            successor = deepcopy(random_walk[-1])

            target_idx = random.randint(0, len(successor.gates) - 1)

            new_gate = random_gate(
                qubit_num, GateSet=CLIFFORD_PLUS_T, weight_types_equally=True
            )

            successor.gates[target_idx] = new_gate

            random_walk.append(successor)
            random_walk_gates.append(get_gate_type(new_gate))

        fitness_scores = fitness.score(random_walk)

        all_fitness_scores.append(fitness_scores)
        all_gate_types.append(random_walk_gates)

    # Compute and plot autocorrelations
    autocorrelations = []

    for offset in range(1, max_offset + 1):
        fitness_scores1 = []
        fitness_scores2 = []

        for fitness_scores in all_fitness_scores:
            fitness_scores1.extend(
                fitness_scores[:len(fitness_scores)-offset]
            )
            fitness_scores2.extend(fitness_scores[offset:])

        autocorrelation = float(stats.pearsonr(
            fitness_scores1, fitness_scores2).correlation)

        autocorrelations.append(autocorrelation)

    ax = plt.subplot()
    ax.plot(range(1, max_offset + 1), autocorrelations)
    ax.set_xlabel("offset")
    ax.set_ylabel("correlation")
    ax.set_ylim(0)

    plt.xticks(list(range(1, max_offset + 1)))
    plt.grid()

    plt.savefig("results/autocorrelation.png", bbox_inches='tight')
    plt.clf()

    # Compute correlation@1 for different gate types

    h_scores1, h_scores2 = [], []
    s_scores1, s_scores2 = [], []
    t_scores1, t_scores2 = [], []
    cx_scores1, cx_scores2 = [], []

    for walk_i in range(len(all_gate_types)):
        for i in range(1, walk_length):
            if all_gate_types[walk_i][i] == "H":
                h_scores1.append(all_fitness_scores[walk_i][i - 1])
                h_scores2.append(all_fitness_scores[walk_i][i])
            elif all_gate_types[walk_i][i] == "S":
                s_scores1.append(all_fitness_scores[walk_i][i - 1])
                s_scores2.append(all_fitness_scores[walk_i][i])
            elif all_gate_types[walk_i][i] == "T":
                t_scores1.append(all_fitness_scores[walk_i][i - 1])
                t_scores2.append(all_fitness_scores[walk_i][i])
            elif all_gate_types[walk_i][i] == "CX":
                cx_scores1.append(all_fitness_scores[walk_i][i - 1])
                cx_scores2.append(all_fitness_scores[walk_i][i])
            else:
                print(f"Warning: Unhandled gate type '{all_gate_types[i]}'")

    h_autocorrelation = float(stats.pearsonr(
        h_scores1, h_scores2).correlation)
    h_correlation_length = compute_correlation_length(h_autocorrelation)
    print(f"H autocorrelation: {h_autocorrelation}")
    print(f"  correlation length: {h_correlation_length}")

    s_autocorrelation = float(stats.pearsonr(
        s_scores1, s_scores2).correlation)
    s_correlation_length = compute_correlation_length(s_autocorrelation)
    print(f"S autocorrelation: {s_autocorrelation}")
    print(f"  correlation length: {s_correlation_length}")

    t_autocorrelation = float(stats.pearsonr(
        t_scores1, t_scores2).correlation)
    t_correlation_length = compute_correlation_length(t_autocorrelation)
    print(f"T autocorrelation: {t_autocorrelation}")
    print(f"  correlation length: {t_correlation_length}")

    cx_autocorrelation = float(stats.pearsonr(
        cx_scores1, cx_scores2).correlation)
    cx_correlation_length = compute_correlation_length(cx_autocorrelation)
    print(f"CX autocorrelation: {cx_autocorrelation}")
    print(f"  correlation length: {cx_correlation_length}")
