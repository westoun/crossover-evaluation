seed_num = 15
seed_offset = 0

mutation_prob = 0.04
crossover_prob = 0.7

generations = 100_000

with open("run_experiments.sh", "w") as target_file:

    for seed_i in range(seed_num):
        seed = seed_offset + seed_i

        for crossover in ["one-point", "pseudo"]:

            experiment_cmd = f"sbatch --job-name=exp_{mutation_prob}mp{crossover_prob}cp_{crossover}_{seed}s"
            experiment_cmd += (f" --export=crossover=\"{crossover}\",mutation_prob={mutation_prob},crossover_prob={crossover_prob},"
                               f"seed={seed},result_dir=/home/ws/ws16/CEIQ/results,generations={generations},target=qft,tag=experiment")
            experiment_cmd += f" run_experiment.sh\n"

            target_file.write(experiment_cmd)

        # Random search
        experiment_cmd = f"sbatch --job-name=exp_1.0mp0.0cp_one-point_{seed}s"
        experiment_cmd += (f" --export=crossover=\"one-point\",mutation_prob=1.0,crossover_prob=0.0,"
                            f"seed={seed},result_dir=/home/ws/ws16/CEIQ/results,generations={generations},target=qft,tag=experiment")
        experiment_cmd += f" run_experiment.sh\n"

        target_file.write(experiment_cmd)

        # Mutation only
        experiment_cmd = f"sbatch --job-name=exp_0.4mp0.0cp_one-point_{seed}s"
        experiment_cmd += (f" --export=crossover=\"one-point\",mutation_prob=0.4,crossover_prob=0.0,"
                            f"seed={seed},result_dir=/home/ws/ws16/CEIQ/results,generations={generations},target=qft,tag=experiment")
        experiment_cmd += f" run_experiment.sh\n"

        target_file.write(experiment_cmd)