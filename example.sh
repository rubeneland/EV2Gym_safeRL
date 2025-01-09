#!/bin/bash
#SBATCH --job-name="10cs_cost_lim_60_epochs_1000"
#SBATCH --partition=compute
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-ee

module load 2024r1 openmpi miniconda3 py-pip

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate EV2Gym_srl

srun python train_safe_RL.py --train cvpo --cost_limit 70 --epoch 1000