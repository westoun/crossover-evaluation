
from datetime import datetime
import json
import os
from quasim import Circuit
from typing import List

from core.params import ExperimentParams
from core.gate_sets import gateset_to_string


def save_to_json(obj, path: str) -> None:
    with open(path, "w") as config_file:
        json.dump(obj, config_file)


def get_timestamp() -> str:
    return str(datetime.now())


def log_params(params: ExperimentParams) -> None:
    target_path = f"{params.result_dir}/"
    if params.tag is not None:
        target_path += f"{params.tag}_"
    target_path += f"{params.qubit_num}q{params.gate_count}g{params.mutation_prob}mp{params.crossover_prob}cp"
    target_path += f"_{params.crossover.name.lower()}"
    target_path += f"_{params.seed}s"
    target_path += f"_config.json"

    config = {
        "start": get_timestamp(),
        "qubit_num": params.qubit_num,
        "gate_count": params.gate_count,
        "gate_set": gateset_to_string(params.gate_set),
        "seed": params.seed,
        "population_size": params.population_size,
        "max_generations": params.max_generations,
        "mutation_prob": params.mutation_prob,
        "crossover_prob": params.crossover_prob,
        "mutation": params.mutation.name,
        "crossover": params.crossover.name,
        "fitness": params.fitness.name,
        "selection": params.selection.name,
        "tag": params.tag
    }

    save_to_json(config, target_path)


def log_fitness(generation: int, best_fitness: float, mean_fitness: float, fitness_stdev: float, top_20_circuits: List[Circuit], params: ExperimentParams) -> None:
    target_path = f"{params.result_dir}/"
    if params.tag is not None:
        target_path += f"{params.tag}_"
    target_path += f"{params.qubit_num}q{params.gate_count}g{params.mutation_prob}mp{params.crossover_prob}cp"
    target_path += f"_{params.crossover.name.lower()}"
    target_path += f"_{params.seed}s"
    target_path += f"_fitness.csv"

    add_header = not os.path.exists(target_path)

    with open(target_path, "a") as target_file:
        if add_header:
            target_file.write(
                f"generation; best fitness; mean fitness; fitness stdev; top 20 circuits;\n")

        top_20_circuits = [circuit.__repr__() for circuit in top_20_circuits]

        target_file.write(
            f"{generation}; {best_fitness}; {mean_fitness}; {fitness_stdev}; {top_20_circuits}\n")
