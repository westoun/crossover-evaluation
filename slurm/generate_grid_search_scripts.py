seed_num = 30
seed_offset = 0

crossover="one-point"

with open("run_grid_search.sh", "w") as target_file:

    for seed_i in range(seed_num):
        seed = seed_offset + seed_i

        for mutation_prob in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
            for crossover_prob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

                experiment_cmd = f"sbatch --job-name=gs_{mutation_prob}mp{crossover_prob}cp_{crossover}_{seed}s"
                experiment_cmd += (f" --export=crossover=\"{crossover}\",mutation_prob={mutation_prob},crossover_prob={crossover_prob},"
                    f"seed={seed},result_dir=/home/ws/ws16/CEIQ/results,generations=1000,target=random,tag=grid_search")
                experiment_cmd += f" run_experiment.sh\n"
                

                target_file.write(experiment_cmd)
