from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import List, Dict


def load_json(path: str) -> Dict:
    with open(path, "r") as json_file:
        return json.load(json_file)


@dataclass
class Experiment:
    qubit_num: int
    gate_count: int
    gate_set: str
    population_size: int
    max_generations: int
    mutation_prob: float
    crossover_prob: float
    mutation: str
    crossover: str
    fitness: str
    selection: str

    seeds: List[int]
    fitness_scores: pd.DataFrame
    fitness_scores_per_seed: List[pd.DataFrame]

    def __hash__(self) -> str:
        key = f"{self.qubit_num}q{self.gate_count}g{self.gate_set}gs"
        key += f"{self.population_size}p{self.max_generations}mg{self.mutation_prob}mp"
        key += f"{self.crossover_prob}cp{self.mutation}m{self.crossover}c"
        key += f"{self.fitness}f{self.selection}s"
        return hash(key)

    def __eq__(self, value):
        return self.__hash__() == value.__hash__()


def load_experiments(results_dir: str = "results") -> List[Experiment]:
    experiments: Dict[str, Experiment] = {}

    for file_name in os.listdir(results_dir):
        if not file_name.endswith("_config.json"):
            continue

        config_path = f"{results_dir}/{file_name}"
        config = load_json(config_path)

        fitness_path = f"{results_dir}/{file_name.replace('_config.json', '_fitness.csv')}"
        experiment_fitness = pd.read_csv(fitness_path, delimiter=";")
        for column in experiment_fitness.columns:
            experiment_fitness = experiment_fitness.rename(columns={
                column: column.strip().replace(" ", "_")
            })

        experiment = Experiment(
            qubit_num=config["qubit_num"],
            gate_count=config["gate_count"],
            gate_set=config["gate_set"],
            population_size=config["population_size"],
            max_generations=config["max_generations"],
            mutation_prob=config["mutation_prob"],
            crossover_prob=config["crossover_prob"],
            mutation=config["mutation"],
            crossover=config["crossover"],
            fitness=config["fitness"],
            selection=config["selection"],

            seeds=[
                config["seed"]
            ],
            fitness_scores=None,
            fitness_scores_per_seed=[experiment_fitness]
        )

        if experiment in experiments:
            experiments[experiment].seeds.append(experiment.seeds[0])
            experiments[experiment].fitness_scores_per_seed.append(
                experiment.fitness_scores_per_seed[0])
        else:
            experiments[experiment] = experiment

    experiments: List[Experiment] = list(experiments.values())
    for experiment in experiments:
        experiment.fitness_scores = merge_dfs(
            experiment.fitness_scores_per_seed)
    return experiments


def merge_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    lengths = [len(df) for df in dfs]
    assert len(set(lengths)) == 1, "Received dataframes with different lengths."

    total_df = pd.concat(dfs)

    avg_best = total_df.groupby("generation")["best_fitness"].mean()
    std_best = total_df.groupby("generation")["best_fitness"].std()
    avg_mean = total_df.groupby("generation")["mean_fitness"].mean()
    std_mean = total_df.groupby("generation")["mean_fitness"].std()

    merged_df = pd.concat([avg_best, std_best, avg_mean, std_mean], keys=[
                          "avg_best", "std_best", "avg_mean", "std_mean"], axis=1)
    merged_df = merged_df.reset_index()
    merged_df.fillna(0, inplace=True)

    return merged_df


def plot_best_fitness_scores(experiments: List[Experiment], target_path: str) -> None:
    ax = plt.subplot()

    # Avoid repeating colors.
    # Source: https://stackoverflow.com/questions/53199728/how-can-i-stop-matplotlib-from-repeating-colors
    ax.set_prop_cycle('color', [plt.get_cmap('gist_rainbow')(
        1.*i/len(experiments)) for i in range(len(experiments))])

    for experiment in experiments:
        label = f"mut_prob={experiment.mutation_prob}, cross_prob={experiment.crossover_prob}"

        df = experiment.fitness_scores

        ax.plot(df["generation"], df["avg_best"], label=label)

    ax.set_ylim(0)

    ax.set_xlabel("generation")
    ax.set_ylabel("avg. best fitness")

    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(target_path, bbox_inches='tight')
    plt.clf()


def plot_mean_fitness_scores(experiments: List[Experiment], target_path: str) -> None:
    ax = plt.subplot()

    # Avoid repeating colors.
    # Source: https://stackoverflow.com/questions/53199728/how-can-i-stop-matplotlib-from-repeating-colors
    ax.set_prop_cycle('color', [plt.get_cmap('gist_rainbow')(
        1.*i/len(experiments)) for i in range(len(experiments))])

    for experiment in experiments:
        label = f"mut_prob={experiment.mutation_prob}, cross_prob={experiment.crossover_prob}"

        df = experiment.fitness_scores

        ax.plot(df["generation"], df["avg_mean"], label=label)

    ax.set_ylim(0)

    ax.set_xlabel("generation")
    ax.set_ylabel("avg. mean fitness")

    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(target_path, bbox_inches='tight')
    plt.clf()


def plot_grid_best_fitness(experiments: List[Experiment], target_path: str) -> None:
    x = []
    y = []
    final_fitness = []

    for experiment in experiments:
        x.append(experiment.crossover_prob)
        y.append(experiment.mutation_prob)
        final_fitness.append(experiment.fitness_scores.iloc[-1]["avg_best"])

    plt.scatter(x, y, c=final_fitness, cmap="magma_r")

    plt.xlabel("crossover probability")
    plt.ylabel("mutation probability")

    plt.ylim(0)

    plt.colorbar()
    plt.grid()

    plt.savefig(target_path, bbox_inches='tight')
    plt.clf()


def plot_grid_mean_fitness(experiments: List[Experiment], target_path: str) -> None:
    x = []
    y = []
    final_fitness = []

    for experiment in experiments:
        x.append(experiment.crossover_prob)
        y.append(experiment.mutation_prob)
        final_fitness.append(experiment.fitness_scores.iloc[-1]["avg_mean"])

    plt.scatter(x, y, c=final_fitness, cmap="magma_r")

    plt.xlabel("crossover probability")
    plt.ylabel("mutation probability")

    plt.ylim(0)

    plt.colorbar()
    plt.grid()

    plt.savefig(target_path, bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":

    experiments = load_experiments(results_dir="results")

    plot_best_fitness_scores(
        experiments, target_path="results/best_fitness_per_generation.png")
    plot_mean_fitness_scores(
        experiments, target_path="results/mean_fitness_per_generation.png")

    plot_grid_best_fitness(
        experiments, target_path="results/best_fitness_on_grid.png")
    plot_grid_mean_fitness(
        experiments, target_path="results/mean_fitness_on_grid.png")
