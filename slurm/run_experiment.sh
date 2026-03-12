#!/bin/bash
#SBATCH --job-name=qcs_gs
#SBATCH --output=logs/slurm.%j.%x.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=10-00:00:00

echo "Activate Python and virtualenv"
cd /home/ws/ws16/CEIQ/

module load python/3.11.1

echo "Call Grid Search Experiment Run"

# Activate virtual environment
source venv/bin/activate

# Run the Python script
echo main.py -c ${crossover} -mp ${mutation_prob} -cp ${crossover_prob} -s ${seed} --result-dir ${result_dir} --generations ${generations} --target ${target} --tag ${tag}
python main.py -c ${crossover} -mp ${mutation_prob} -cp ${crossover_prob} -s ${seed} --result-dir ${result_dir} --generations ${generations} --target ${target} --tag ${tag}

