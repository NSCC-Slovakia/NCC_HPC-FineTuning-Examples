#!/bin/bash
#SBATCH --account=<your_project_number>
#SBATCH -o result.txt
#SBATCH -e error.txt
#SBATCH --gres=gpu:4 # Number of GPUs(1 to 4)
#SBATCH --nodes=2
#SBATCH --partition=gpu

module load singularity

# Include commands in output:
set -x

# Print current time and date:
date

# Print host name:
hostname

# List available GPUs:
nvidia-smi

# Set environment variables for communication between nodes:
export MASTER_PORT=24998
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)

# Run AI scripts:
# time conda run -n finetuning --no-capture-output torchrun --nproc_per_node 2 mistral7b_train_ddp.py
time srun conda run -n finetuning --no-capture-output torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d \
    singularity exec --nv /storage-data/singularity_containers/pt-2.3_llm.sif python3 train_mistral_peft.py 