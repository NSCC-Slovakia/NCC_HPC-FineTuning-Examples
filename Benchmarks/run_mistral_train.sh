#!/bin/bash
#SBATCH --account= # Project number
#SBATCH --job-name= # Name of the job in SLURM
#SBATCH --partition=gpu # Selected partition
#SBATCH --gres=gpu:2 # Number of GPUs
#SBATCH --nodes=1 # Number of nodes
##SBATCH --ntasks-per-node= # Number of MPI ranks
#SBATCH --cpus-per-task=14# Number of OMP threads per MPI rank
#SBATCH --output job%J.out # Standard output
#SBATCH --error job%J.err # Standard error

# Load modules
module purge
module load singularity
# Disable users python packages
export PYTHONNOUSERSITE=1

# Execute train_mistral_peft.py
singularity exec --nv /home/valabek/HPC-FineTuning-Examples/pytorch_updated.sif torchrun --nproc_per_node 2 mistral7b_train.py 

