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
from core.gate_sets import CLIFFORD_PLUS_T
from core.utils.random_ import random_circuit


if __name__ == "__main__":
    gate_count = 10
    qubit_num = 3
    population_size = 1000
    max_generations = 100

    seed_num = 5
    seed_offset = 0

    mutation = ReplaceGateMutation(
        qubit_num=qubit_num, gate_set=CLIFFORD_PLUS_T
    )

    hc_crossover = HeadlessChickenCrossover(
        crossover=OnePointCrossover(),
        qubit_num=qubit_num,
        gate_count=gate_count,
        gate_set=CLIFFORD_PLUS_T
    )
    op_crossover = OnePointCrossover()

    for seed_i in tqdm(range(seed_num), desc="Seeds"):
        seed = seed_offset + seed_i

        for mutation_prob in tqdm([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], leave=False, desc="Mut Probs"):
            for crossover_prob in tqdm([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], leave=False, desc="Cross Probs"):

                for crossover in [
                    op_crossover, hc_crossover
                ]:

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
                        gate_set=CLIFFORD_PLUS_T,
                        seed=seed,
                        result_dir="results",
                        tag="gs")

                    ga = GeneticAlgorithm(params)
                    ga.run()
