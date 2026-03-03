from copy import deepcopy
from dataclasses import dataclass
import matplotlib.pyplot as plt
from numpy import random as np_random
from quasim import Circuit, get_unitary
from quasim.gates import IGate
import random
from scipy import stats
from statistics import mean, stdev
from tqdm import tqdm
from typing import List
import warnings

from core.utils.random_ import random_circuit, random_gate
from core.utils.logging import save_to_json, get_timestamp
from core.fitness import Fitness, AbsoluteDistanceFitness
from core.gate_sets import CLIFFORD_PLUS_T


class AppendGateMutation:
    name: str = "Append Gate Mutation"

    qubit_num: int
    gate_set: List[IGate]

    def __init__(self, qubit_num: int, gate_set: List[IGate]):
        self.qubit_num = qubit_num
        self.gate_set = gate_set

    def mutate(self, circuit: Circuit) -> Circuit:
        circuit = deepcopy(circuit)
        new_gate = random_gate(
            self.qubit_num, self.gate_set)

        circuit.gates.append(new_gate)
        return circuit


class InsertSomewhereMutation:
    name: str = "Insert Somewhere Mutation"

    qubit_num: int
    gate_set: List[IGate]

    def __init__(self, qubit_num: int, gate_set: List[IGate]):
        self.qubit_num = qubit_num
        self.gate_set = gate_set

    def mutate(self, circuit: Circuit) -> Circuit:
        circuit = deepcopy(circuit)

        new_gate = random_gate(
            self.qubit_num, self.gate_set)

        target_i = random.randint(0, len(circuit.gates))
        circuit.gates.insert(
            target_i, new_gate
        )

        return circuit


@dataclass
class MutationResult():
    fitness_before: float
    fitness_after: float
    gate_type: str
    gates_before: int


def get_gate_type(gate: IGate) -> str:
    return str(type(gate)).split(".")[-1].split("'")[0]


def get_differing_gate(circuit1: Circuit, circuit2: Circuit) -> IGate:

    assert abs(len(circuit1.gates) - len(circuit2.gates)) == 1

    if len(circuit1.gates) > len(circuit2.gates):
        longer_circuit = circuit1
        shorter_circuit = circuit2
    else:
        longer_circuit = circuit2
        shorter_circuit = circuit1

    for i in range(len(shorter_circuit.gates)):
        if shorter_circuit.gates[i].__repr__() != longer_circuit.gates[i].__repr__():
            return longer_circuit.gates[i]

    return longer_circuit.gates[-1]


if __name__ == "__main__":

    seed_num = 100
    population_size = 200
    min_gates = 1
    max_gates = 20
    qubit_num = 4

    mutation = AppendGateMutation(qubit_num, gate_set=CLIFFORD_PLUS_T)
    # mutation = InsertSomewhereMutation(qubit_num, gate_set=CLIFFORD_PLUS_T)

    data: List[MutationResult] = []
    for seed in tqdm(range(seed_num), total=seed_num):

        random.seed(seed)
        np_random.seed(seed)

        target_circuit: Circuit = random_circuit(
            qubit_num=qubit_num, gate_count=random.randint(min_gates, max_gates), gate_set=CLIFFORD_PLUS_T
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target_unitary = get_unitary(target_circuit)

        fitness = AbsoluteDistanceFitness(
            target_unitary=target_unitary
        )

        old_population = [
            random_circuit(
                qubit_num=qubit_num, gate_count=min_gates, gate_set=CLIFFORD_PLUS_T
            ) for _ in range(population_size)
        ]
        old_scores = fitness.score(old_population)

        for gates_before in range(min_gates, max_gates):

            new_population = [
                mutation.mutate(circuit) for circuit in old_population
            ]
            new_scores = fitness.score(new_population)

            for circuit, old_circuit, score, old_score in zip(new_population, old_population, new_scores, old_scores):

                differing_gate = circuit.gates[-1]
                # differing_gate = get_differing_gate(circuit, old_circuit)

                data.append(
                    MutationResult(
                        fitness_before=old_score, fitness_after=score, gate_type=get_gate_type(differing_gate), gates_before=gates_before
                    )
                )

            old_population = new_population
            old_scores = new_scores

    # show gates before on x axis, autocorrelation for current gate count on y, all gate types

    gate_types = list(set([
        datum.gate_type for datum in data
    ]))
    gate_types = sorted(gate_types)

    ax = plt.subplot()

    for gate_type in gate_types:
        gate_type_data = [
            datum for datum in data if datum.gate_type == gate_type
        ]

        circuit_lengths = []
        autocorrelations = []

        for gates_before in range(min_gates, max_gates):

            rel_data = [
                datum for datum in gate_type_data if datum.gates_before == gates_before
            ]

            circuit_lengths.append(gates_before)

            fitness_scores_before = [
                datum.fitness_before for datum in rel_data
            ]
            fitness_scores_after = [
                datum.fitness_after for datum in rel_data
            ]

            autocorrelation = stats.pearsonr(
                fitness_scores_before, fitness_scores_after).correlation
            autocorrelations.append(autocorrelation)

        ax.plot(circuit_lengths, autocorrelations, label=gate_type)

    # general autocorrelations
    circuit_lengths = []
    autocorrelations = []

    for gates_before in range(min_gates, max_gates):
        rel_data = [
            datum for datum in data if datum.gates_before == gates_before
        ]

        circuit_lengths.append(gates_before)

        fitness_scores_before = [
            datum.fitness_before for datum in rel_data
        ]
        fitness_scores_after = [
            datum.fitness_after for datum in rel_data
        ]

        autocorrelation = stats.pearsonr(
            fitness_scores_before, fitness_scores_after).correlation
        autocorrelations.append(autocorrelation)

    ax.plot(circuit_lengths, autocorrelations,
            label="All Gates", color="grey", linestyle="dashed")

    ax.set_ylabel("autocorrelation")
    ax.set_xlabel("circuit length")

    ax.set_xticks(range(min_gates, max_gates))

    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(
        f"results/autocorrelations_append_gate_{qubit_num}q.png", bbox_inches='tight')
    plt.show()

    plt.clf()
