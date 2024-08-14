module load singularity
time singularity exec --nv /storage-data/singularity_containers/pt-2.3_llm.sif python3 train_peft.py
