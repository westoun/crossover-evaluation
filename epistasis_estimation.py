import matplotlib.pyplot as plt
from numpy import random as np_random
import pandas as pd
from quasim import Circuit, get_unitary
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as compute_r2_score
from statistics import mean, stdev, median
from tqdm import tqdm
import warnings

from core.utils.random_ import random_circuit
from core.gate_sets import CLIFFORD_PLUS_T, CLIFFORD_PLUS_T_PLUS_I
from core.fitness import Fitness, AbsoluteDistanceFitness


def estimate_r2_score(qubit_num: int, gate_count: int, population_size: int, seed: int = 0) -> float:
    random.seed(seed)
    np_random.seed(seed)

    target_circuit: Circuit = random_circuit(
        qubit_num=qubit_num, gate_count=gate_count, gate_set=gate_set
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target_unitary = get_unitary(target_circuit)

    fitness = AbsoluteDistanceFitness(
        target_unitary=target_unitary
    )

    circuits = [
        random_circuit(qubit_num, gate_count, gate_set) for _ in range(population_size)
    ]

    fitness_scores = fitness.score(circuits)

    data = []
    for circuit in circuits:
        data.append([])
        for gate in circuit.gates:
            data[-1].append(gate.__repr__())

    df = pd.DataFrame(data)
    X = pd.get_dummies(data=df, drop_first=True)
    y = pd.DataFrame(fitness_scores)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regressor = LinearRegression().fit(X, y)
        y_pred = regressor.predict(X)

    r2_score = compute_r2_score(y_true=y, y_pred=y_pred)
    return r2_score


if __name__ == "__main__":

    seed_num = 30
    population_size = 10_000

    gate_set = CLIFFORD_PLUS_T

    ax = plt.subplot()

    qubit_nums = [2, 4, 6]
    gate_counts = [1, 2, 3, 4, 5 , 6, 7, 8, 9, 10]

    for qubit_num in tqdm(qubit_nums, desc="Qubits"):

        r2_scores_per_gate_count = []

        for gate_count in tqdm(gate_counts, desc="Gates", leave=False):

            r2_scores = []
            for seed in tqdm(range(seed_num), total=seed_num, leave=False):
                r2_score = estimate_r2_score(
                    qubit_num=qubit_num, gate_count=gate_count, population_size=population_size, seed=seed)
                r2_scores.append(r2_score)

            r2_scores_per_gate_count.append(median(r2_scores))

        ax.plot(gate_counts, r2_scores_per_gate_count,
                label=f"{qubit_num} qubits", linewidth=1)

    plt.xticks(gate_counts)
    plt.xlabel("gate count")

    plt.ylabel("r2")
    plt.ylim(0)

    plt.grid()
    plt.legend()

    plt.savefig(
        f"results/epistasis.png", bbox_inches='tight')
    plt.clf()
