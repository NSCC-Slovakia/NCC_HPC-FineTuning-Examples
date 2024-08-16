#!/bin/bash
#SBATCH --account=<your_project_number>
#SBATCH -o result.txt
#SBATCH -e error.txt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=gpu

module load singularity

singularity exec --nv /storage-data/singularity_containers/pt-2.3_llm.sif python3 mistral_trained_pred.py 

