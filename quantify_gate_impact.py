from copy import deepcopy
import editdistance
from numpy import random as np_random
from quasim import Circuit, get_unitary
from quasim.gates import IGate
import random
from statistics import mean, stdev
from tqdm import tqdm
from typing import List
import warnings

from core.utils.random_ import random_circuit
from core.utils.logging import save_to_json, get_timestamp
from core.fitness import Fitness, AbsoluteDistanceFitness
from core.gate_sets import CLIFFORD_PLUS_T


def get_gate_type(gate: IGate) -> str:
    return str(type(gate)).split(".")[-1].split("'")[0]


if __name__ == "__main__":
    qubit_num = 2
    gate_count = 10
    circuit_count = 5

    random.seed(0)
    np_random.seed(0)

    for _ in range(circuit_count):

        target_circuit: Circuit = random_circuit(
            qubit_num=qubit_num, gate_count=gate_count, gate_set=CLIFFORD_PLUS_T
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target_unitary = get_unitary(target_circuit)

        fitness = AbsoluteDistanceFitness(
            target_unitary=target_unitary
        )

        ablation_circuits = []
        for gate_i in range(gate_count):
            ablation_circuit = deepcopy(target_circuit)
            ablation_circuit.gates.pop(gate_i)
            ablation_circuits.append(ablation_circuit)

        fitness_scores = fitness.score(ablation_circuits)
        fitness_scores = [
            score / (2**(1.5 * qubit_num + 1)) for score in fitness_scores
        ]

        print("")
        for gate, fitness_score in zip(target_circuit.gates, fitness_scores):
            print(f"{gate}: {fitness_score}")

        print("")
        for gate_type in ["H", "CX", "S", "T"]:
            ablation_circuit = deepcopy(target_circuit)
            for i in range(len(ablation_circuit.gates)):
                gate_i = len(ablation_circuit.gates) - i - 1

                if get_gate_type(ablation_circuit.gates[gate_i]).split("(")[0] == gate_type:
                    ablation_circuit.gates.pop(gate_i)

            fitness_score = fitness.score([ablation_circuit])[0] / (2**(1.5 * qubit_num + 1))
            print(f"Change if all {gate_type} gates were removed: {fitness_score}")
