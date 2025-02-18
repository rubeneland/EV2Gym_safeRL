#!/bin/bash
#SBATCH --job-name="exp1"
#SBATCH --partition=compute
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --account=education-eemcs-msc-ee

module load 2024r1 openmpi miniconda3 py-pip

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate EV2Gym_srl

srun python train_safe_RL.py --train cvpo --cost_limit 1 --epoch 900 --train_num 4 --test_num 50