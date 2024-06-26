#!/bin/bash
#SBATCH --job-name=ImBaby
#SBATCH --nodes=1
#SBATCH --partition=it-hpc
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=10G
#SBATCH --time=12:00:00

module purge

python cdnod.py

