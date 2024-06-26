#!/bin/bash
#SBATCH --job-name=notebook
#SBATCH --nodes=1
#SBATCH --partition=it-hpc
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=20G
#SBATCH --time=12:00:00

module purge
source ~/.bashrc
cd ~

port=54376

# jupyter notebook --no-browser --port=$port

# in the login node, enter " ssh -f -NB -L 54376:localhost:54376 mignhao.fu@mbzuai.ac.ae@gpu40-3 "