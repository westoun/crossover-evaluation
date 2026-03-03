from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats.stats import pearsonr
from statistics import mean
from typing import List, Dict


def load_json(path: str) -> Dict:
    with open(path, "r") as json_file:
        return json.load(json_file)


def flatten_child_fitness_scores(pairing: Dict) -> List[float]:
    child_fitness_scores = []

    for child_scores in pairing["child_fitness"]:
        child_fitness_scores.extend(child_scores)

    return child_fitness_scores


def plot_grid_as_landscape(
        parent1_fitness: List[float], parent2_fitness: List[float], child_fitness: List[float]
):
    ax = plt.axes(projection="3d")
    ax.plot_trisurf(parent1_fitness, parent2_fitness,
                    child_fitness, cmap="magma_r", linewidth=0.2)

    ax.set_xlabel("parent 1 fitness")
    ax.set_ylabel("parent 2 fitness")
    ax.set_zlabel("max child fitness")
    ax.set_xlim(min(parent1_fitness), max(parent1_fitness))
    ax.set_ylim(min(parent2_fitness), max(parent2_fitness))
    ax.invert_zaxis()

    # plt.savefig(target_path, bbox_inches='tight')

    plt.show()
    plt.clf()


def plot_grid_as_scatter(parent1_fitness: List[float], parent2_fitness: List[float], child_fitness: List[float], target_path: str = None, show: bool = False) -> None:
    plt.scatter(parent1_fitness, parent2_fitness,
                c=child_fitness, cmap="magma_r", s=1)

    plt.xlabel("parent 1 fitness")
    plt.ylabel("parent 2 fitness")

    plt.ylim(0, max(parent1_fitness + parent2_fitness))
    plt.xlim(0, max(parent1_fitness + parent2_fitness))

    plt.colorbar()
    plt.grid()

    if target_path is not None:
        plt.savefig(target_path, bbox_inches='tight')

    if show:
        plt.show()

    plt.clf()


if __name__ == "__main__":

    file_names = os.listdir("results")
    file_names.sort()

    for file_name in file_names:
        if not (file_name.startswith("guided_walk") and file_name.endswith(".json")):
            continue

        experiment = load_json(f"results/{file_name}")

        print(
            f"\nStarting evaluation of experiment with {experiment['config']['qubit_num']}q and {experiment['config']['gate_count']}g")

        parent1_fitness = []
        parent2_fitness = []
        min_parent_fitness = []
        max_parent_fitness = []
        mean_parent_fitness = []
        parent_fitness_span = []

        min_child_fitness = []
        mean_child_fitness = []

        improvement_percentage = []

        for pairing in experiment["results"]:
            parent1_fitness.append(pairing["parent_fitness"][0])
            parent2_fitness.append(pairing["parent_fitness"][1])
            min_parent_fitness.append(min(pairing["parent_fitness"]))
            max_parent_fitness.append(max(pairing["parent_fitness"]))
            mean_parent_fitness.append(mean(pairing["parent_fitness"]))

            parent_fitness_span.append(
                abs(pairing["parent_fitness"][0] - pairing["parent_fitness"][1]))

            child_fitness_scores = flatten_child_fitness_scores(pairing)
            min_child_fitness.append(
                min(child_fitness_scores)
            )
            mean_child_fitness.append(
                mean(child_fitness_scores)
            )

            improvement_percentage.append(
                len([
                    child_score for child_score in child_fitness_scores if child_score < min(pairing["parent_fitness"])
                ]) / len(child_fitness_scores)
            )

        distance_to_parent = [
            (child - parent) for child, parent in zip(min_child_fitness, min_parent_fitness)
        ]

        print(f"\tAverage percentage of children better than best parent: {mean(improvement_percentage)}")
        print(f"\tAverage fitness of best child: {mean(min_child_fitness)}")

        print(
            f"\tCorrelation best parent - distance to best child: {pearsonr(min_parent_fitness, distance_to_parent)}")

        print("")
        print(
            f"\tCorrelation second parent - best child: {pearsonr(parent2_fitness, min_child_fitness)}")
        print(
            f"\tCorrelation second parent - mean child: {pearsonr(parent2_fitness, mean_child_fitness)}")
        print(
            f"\tCorrelation best parent - best child: {pearsonr(min_parent_fitness, min_child_fitness)}")
        print(
            f"\tCorrelation best parent - mean child: {pearsonr(min_parent_fitness, mean_child_fitness)}")
        print(
            f"\tCorrelation worst parent - best child: {pearsonr(max_parent_fitness, min_child_fitness)}")

        print("")
        print(
            f"\tCorrelation best parent - percentage of improvements: {pearsonr(min_parent_fitness, improvement_percentage)}")
        print(
            f"\tCorrelation second parent - percentage of improvements: {pearsonr(parent2_fitness, improvement_percentage)}")
        print(
            f"\tCorrelation parent spread - percentage of improvements: {pearsonr(parent_fitness_span, improvement_percentage)}")

        plot_grid_as_scatter(parent1_fitness, parent2_fitness, min_child_fitness,
                             target_path=f"results/guided_walk_{experiment['config']['qubit_num']}q{experiment['config']['gate_count']}g_best.png")
        plot_grid_as_scatter(parent1_fitness, parent2_fitness, mean_child_fitness,
                             target_path=f"results/guided_walk_{experiment['config']['qubit_num']}q{experiment['config']['gate_count']}g_mean.png")
