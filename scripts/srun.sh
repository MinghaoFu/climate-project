#!/bin/bash

#SBATCH --job-name=HF_1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=64
#SBATCH --mem=64G

#SBATCH --partition=gpu
#SBATCH --qos=gpu-8

python train_CESM2.py