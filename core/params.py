from dataclasses import dataclass
from quasim.gates import IGate
import random
from typing import List

from .mutation import Mutation
from .crossover import Crossover
from .fitness import Fitness
from .selection import Selection


@dataclass
class ExperimentParams:
    crossover: Crossover
    mutation: Mutation

    mutation_prob: float
    crossover_prob: float

    fitness: Fitness
    selection: Selection

    population_size: int
    max_generations: int

    qubit_num: int
    gate_count: int
    gate_set: List[IGate]

    seed: int
    result_dir: str
    tag: str = None
