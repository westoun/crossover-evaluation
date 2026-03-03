from copy import deepcopy
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

def extract_gate_type_statistics(circuits: List[Circuit]):
    h_gate_counts = []
    s_gate_counts = []
    t_gate_counts = []
    cx_gate_counts = []
    
    for circuit in circuits:
        h_gate_counts.append(0)
        s_gate_counts.append(0)
        t_gate_counts.append(0)
        cx_gate_counts.append(0)

        for gate in circuit.gates:
            gate_type = get_gate_type(gate)

            if gate_type == "H":
                h_gate_counts[-1] += 1 
            elif gate_type == "S":
                s_gate_counts[-1] += 1 
            elif gate_type == "T":
                t_gate_counts[-1] += 1 
            elif gate_type == "CX":
                cx_gate_counts[-1] += 1 
            else:
                raise ValueError(f"Gate type: {gate_type}")

    print(f"\tH count: {mean(h_gate_counts)} (std: {stdev(h_gate_counts)})")
    print(f"\tS count: {mean(s_gate_counts)} (std: {stdev(s_gate_counts)})")
    print(f"\tT count: {mean(t_gate_counts)} (std: {stdev(t_gate_counts)})")
    print(f"\tCX count: {mean(cx_gate_counts)} (std: {stdev(cx_gate_counts)})")

if __name__ == "__main__":
    
    seed_offset = 5
    seed_num = 10
    qubit_num = 3
    gate_count = 10
    population_size = 100_000
    n = 10

    for seed in range(seed_num):

        seed = seed + seed_offset
        print()
        print(f"Results for seed {seed}:")

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
        scores = fitness.score(population)


        circuit_scores = list(zip(population, scores))
        circuit_scores.sort(key=lambda item: item[1])
        print(f"Fitness range: {round(circuit_scores[0][1])} - {round(circuit_scores[-1][1])}")
        
        best_n_circuits = [
            circuit for (circuit, score) in circuit_scores[:n]
        ]
        
        worst_n_circuits = [
            circuit for (circuit, score) in circuit_scores[-n:]
        ]
        print("True circuit:")
        extract_gate_type_statistics([target_circuit, target_circuit])
        print(f"Best circuit:")
        extract_gate_type_statistics([best_n_circuits[0], best_n_circuits[0]])
        print(f"Best circuits:")
        extract_gate_type_statistics(best_n_circuits)
        print(f"Worst circuits:")
        extract_gate_type_statistics(worst_n_circuits)