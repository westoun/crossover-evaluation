from copy import deepcopy
from dataclasses import dataclass
import math
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
from core.mutation import ReplaceGateMutation


@dataclass
class MutationResult():
    gate_count: int
    qubit_num: int
    fitness_before: float
    fitness_after: float
    gate_type: str


def get_gate_type(gate: IGate) -> str:
    return str(type(gate)).split(".")[-1].split("'")[0]


def get_differing_gate(old_circuit: Circuit, new_circuit: Circuit) -> IGate:

    assert len(old_circuit.gates) == len(new_circuit.gates)

    for old_gate, new_gate in zip(old_circuit.gates, new_circuit.gates):
        if old_gate.__repr__() != new_gate.__repr__():
            return new_gate


def filter_results(data: List[MutationResult], criterion: str, value) -> List[MutationResult]:
    if criterion == "qubit_num":
        filtered_data = [
            datum for datum in data if datum.qubit_num == value
        ]
        return filtered_data
    elif criterion == "gate_count":
        filtered_data = [
            datum for datum in data if datum.gate_count == value
        ]
        return filtered_data
    elif criterion == "gate_type":
        filtered_data = [
            datum for datum in data if datum.gate_type == value
        ]
        return filtered_data
    else:
        raise NotImplementedError()


def compute_correlation_length(autocorrelation: float) -> float:
    correlation_length = -1 / math.log(autocorrelation)
    return correlation_length


if __name__ == "__main__":

    seed_num = 100
    population_size = 500

    qubit_nums = [2, 3, 4, 5, 6]
    gate_counts = [10, 15, 20, 25, 30]

    data: List[MutationResult] = []

    for qubit_num in tqdm(qubit_nums, desc="Qubits"):
        for gate_count in tqdm(gate_counts, desc="Gate Counts", leave=False):

            for seed in tqdm(range(seed_num), total=seed_num, desc="Seed", leave=False):

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

                population = [
                    random_circuit(
                        qubit_num=qubit_num, gate_count=gate_count, gate_set=CLIFFORD_PLUS_T
                    ) for _ in range(population_size)
                ]
                old_scores = fitness.score(population)

                for circuit, old_score in zip(
                    population, old_scores
                ):
                    target_idx = random.randint(0, len(circuit.gates) - 1)

                    circuit.gates[target_idx] = random_gate(
                        qubit_num, GateSet=CLIFFORD_PLUS_T, weight_types_equally=True
                    )

                    new_score = fitness.score([circuit])[0]

                    data.append(
                        MutationResult(
                            gate_count=gate_count,
                            qubit_num=qubit_num,
                            fitness_before=old_score,
                            fitness_after=new_score,
                            gate_type=get_gate_type(circuit.gates[target_idx])
                        )
                    )

    gate_types = list(set([
        datum.gate_type for datum in data
    ]))
    gate_types = sorted(gate_types)

    # fixed qubit, varying gate count
    for qubit_num in qubit_nums:

        ax = plt.subplot()

        data_by_qubit = filter_results(
            data, criterion="qubit_num", value=qubit_num
        )

        # autocorrelation per gate type
        for gate_type in gate_types:
            data_by_gate_type = filter_results(
                data_by_qubit, criterion="gate_type", value=gate_type
            )

            autocorrelations = []
            for gate_count in gate_counts:
                data_by_gate_count = filter_results(
                    data_by_gate_type, criterion="gate_count", value=gate_count
                )

                before_values = [
                    datum.fitness_before for datum in data_by_gate_count]
                after_values = [
                    datum.fitness_after for datum in data_by_gate_count]

                autocorrelation = stats.pearsonr(
                    before_values, after_values).correlation
                autocorrelations.append(autocorrelation)

            ax.plot(gate_counts, autocorrelations,
                    label=gate_type)

        # autocorrelation across gate types
        autocorrelations = []
        for gate_count in gate_counts:
            data_by_gate_count = filter_results(
                data_by_qubit, criterion="gate_count", value=gate_count
            )

            before_values = [
                datum.fitness_before for datum in data_by_gate_count]
            after_values = [
                datum.fitness_after for datum in data_by_gate_count]

            autocorrelation = stats.pearsonr(
                before_values, after_values).correlation
            autocorrelations.append(autocorrelation)

        ax.plot(gate_counts, autocorrelations,
                label="All Gates", color="grey", linestyle="dashed")

        ax.set_ylabel("autocorrelation")
        ax.set_xlabel("circuit length")

        ax.set_xticks(gate_counts)

        ax.set_ylim(0)

        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(
            f"results/autocorrelations_{qubit_num}q_varying_gates.png", bbox_inches='tight')
        plt.clf()

    # fixed gate count, varying qubits
    for gate_count in gate_counts:

        ax = plt.subplot()

        data_by_gate_count = filter_results(
            data, criterion="gate_count", value=gate_count
        )

        # autocorrelation per gate type
        for gate_type in gate_types:
            data_by_gate_type = filter_results(
                data_by_gate_count, criterion="gate_type", value=gate_type
            )

            autocorrelations = []
            for qubit_num in qubit_nums:
                data_by_qubit = filter_results(
                    data_by_gate_type, criterion="qubit_num", value=qubit_num
                )

                before_values = [
                    datum.fitness_before for datum in data_by_qubit]
                after_values = [datum.fitness_after for datum in data_by_qubit]

                autocorrelation = stats.pearsonr(
                    before_values, after_values).correlation
                autocorrelations.append(autocorrelation)

            ax.plot(qubit_nums, autocorrelations,
                    label=gate_type)

        # autocorrelation across gate types
        autocorrelations = []
        for qubit_num in qubit_nums:
            data_by_qubit = filter_results(
                data_by_gate_count, criterion="qubit_num", value=qubit_num
            )

            before_values = [datum.fitness_before for datum in data_by_qubit]
            after_values = [datum.fitness_after for datum in data_by_qubit]

            autocorrelation = stats.pearsonr(
                before_values, after_values).correlation
            autocorrelations.append(autocorrelation)

        ax.plot(qubit_nums, autocorrelations,
                label="All Gates", color="grey", linestyle="dashed")

        ax.set_ylabel("autocorrelation")
        ax.set_xlabel("qubit num")

        ax.set_xticks(qubit_nums)

        ax.set_ylim(0)

        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(
            f"results/autocorrelations_{gate_count}g_varying_qubits.png", bbox_inches='tight')
        plt.clf()
