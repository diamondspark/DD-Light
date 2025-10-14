#!/bin/bash
#SBATCH -o logs/cache_7_pgk2_h2o_final_%A_%a.out  
#SBATCH --job-name=cache_7_
#SBATCH --partition=gpu-long
#SBATCH --time=484-24:00:00
#SBATCH --mem=500000M
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dds

# Run the Python script with the seed_idx argument as SLURM_ARRAY_TASK_ID
python -u FinalDocking.py
