
from datetime import datetime
import json
import os

from core.params import ExperimentParams
from core.gate_sets import gateset_to_string


def save_to_json(obj, path: str) -> None:
    with open(path, "w") as config_file:
        json.dump(obj, config_file)


def get_timestamp() -> str:
    return str(datetime.now())


def log_params(params: ExperimentParams) -> None:
    target_path = f"{params.result_dir}/{params.crossover.name.lower()}"
    target_path += f"_{params.qubit_num}q{params.gate_count}g{params.seed}s{params.mutation_prob}mp{params.crossover_prob}cp"
    if params.tag is not None:
        target_path += f"_{params.tag}"
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
    }

    save_to_json(config, target_path)


def log_fitness(generation: int, best_fitness: float, mean_fitness: float, fitness_stdev: float, params: ExperimentParams) -> None:
    target_path = f"{params.result_dir}/{params.crossover.name.lower()}"
    target_path += f"_{params.qubit_num}q{params.gate_count}g{params.seed}s{params.mutation_prob}mp{params.crossover_prob}cp"
    if params.tag is not None:
        target_path += f"_{params.tag}"
    target_path += f"_fitness.csv"

    add_header = not os.path.exists(target_path)

    with open(target_path, "a") as target_file:
        if add_header:
            target_file.write(
                f"generation; best fitness; mean fitness; fitness stdev\n")

        target_file.write(
            f"{generation}; {best_fitness}; {mean_fitness}; {fitness_stdev}\n")
