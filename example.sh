#!/bin/bash
#SBATCH --job-name="allcost_10cs"
#SBATCH --partition=compute
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --account=education-eemcs-msc-ee

module load 2024r1 openmpi miniconda3 py-pip

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate EV2Gym_srl

srun python train_safe_RL.py --train ppol --cost_limit 80 --epoch 600 --train_num 10 --test_num 50