#!/bin/bash
#SBATCH --job-name="20cs_cost_lim_250_epochs_300"
#SBATCH --partition=compute
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-ee

module load 2024r1 openmpi miniconda3 py-pip

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate EV2Gym_srl

srun python train_safe_RL.py --train cvpo --cost_limit 120 --epoch 300