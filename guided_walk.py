from copy import deepcopy
from numpy import random as np_random
from quasim import Circuit, get_unitary
import random
import warnings

from core.utils.random_ import random_circuit
from core.utils.logging import save_to_json, get_timestamp
from core.fitness import Fitness, AbsoluteDistanceFitness
from core.gate_sets import CLIFFORD_PLUS_T


if __name__ == "__main__":

    seed_num = 30
    qubit_num = 3
    gate_count = 10
    pairings_per_seed = 1000

    experiment = {
        "config": {
            "start": get_timestamp(),
            "seed_num": seed_num,
            "qubit_num": qubit_num,
            "gate_count": gate_count,
            "pairings_per_seed": pairings_per_seed
        },
        "results": []
    }

    for seed in range(seed_num):
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

        for pairing in range(pairings_per_seed):
            parent1 = random_circuit(
                qubit_num=qubit_num, gate_count=gate_count, gate_set=CLIFFORD_PLUS_T
            )
            parent2 = random_circuit(
                qubit_num=qubit_num, gate_count=gate_count, gate_set=CLIFFORD_PLUS_T
            )

            parent1_fitness = fitness.score(
                [parent1]
            )[0]
            parent2_fitness = fitness.score(
                [parent2]
            )[0]

            experiment["results"].append(
                {
                    "parent_fitness": (parent1_fitness, parent2_fitness),
                    "child_fitness": []
                }
            )

            for split_point in range(1, gate_count):
                child1 = deepcopy(parent1)
                child2 = deepcopy(parent2)

                child1.gates[split_point:], child2.gates[split_point:] = (
                    deepcopy(child2.gates[split_point:]),
                    deepcopy(child1.gates[split_point:]),
                )

                child1_fitness = fitness.score(
                    [child1]
                )[0]
                child2_fitness = fitness.score(
                    [child2]
                )[0]

                experiment["results"][-1]["child_fitness"].append(
                    (child1_fitness, child2_fitness)
                )

    save_to_json(experiment, "results/guided_walk.json")