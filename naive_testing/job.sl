#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --output=gpujob_%A_%a.out
#SBATCH --gres=gpu:v100:1

module purge
module load nvidia-hpc-sdk
module load gcc/8.3.0

./naive
