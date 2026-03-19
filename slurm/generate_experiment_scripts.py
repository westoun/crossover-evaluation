seed_num = 30
seed_offset = 30

mutation_prob = 0.04
crossover_prob = 0.7

generations = 20_000

# Adjust tag if different target is used.
# target = "qft"
target = "haar-state"
tag = f"{target}_experiment"

with open("slurm/run_experiments.sh", "w") as target_file:

    for seed_i in range(seed_num):
        seed = seed_offset + seed_i

        # One point crossover
        experiment_cmd = f"sbatch --job-name=exp_{mutation_prob}mp{crossover_prob}cp_one-point_{seed}s"
        experiment_cmd += (f" --export=crossover=\"one-point\",mutation_prob={mutation_prob},crossover_prob={crossover_prob},"
                            f"seed={seed},result_dir=/home/ws/ws16/CEIQ/results,generations={generations},target=\"{target}\",tag=\"{tag}\"")
        experiment_cmd += f" run_experiment.sh\n"

        target_file.write(experiment_cmd)

        # Random search
        experiment_cmd = f"sbatch --job-name=exp_1.0mp0.0cp_one-point_{seed}s"
        experiment_cmd += (f" --export=crossover=\"one-point\",mutation_prob=1.0,crossover_prob=0.0,"
                            f"seed={seed},result_dir=/home/ws/ws16/CEIQ/results,generations={generations},target=\"{target}\",tag=\"{tag}\"")
        experiment_cmd += f" run_experiment.sh\n"

        target_file.write(experiment_cmd)

        # Mutation only
        experiment_cmd = f"sbatch --job-name=exp_0.04mp0.0cp_one-point_{seed}s"
        experiment_cmd += (f" --export=crossover=\"one-point\",mutation_prob=0.04,crossover_prob=0.0,"
                            f"seed={seed},result_dir=/home/ws/ws16/CEIQ/results,generations={generations},target=\"{target}\",tag=\"{tag}\"")
        experiment_cmd += f" run_experiment.sh\n"

        target_file.write(experiment_cmd)