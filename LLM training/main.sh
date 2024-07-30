module load singularity
time singularity exec --nv /storage-data/singularity_containers/pytorch-2.1_llm_flash-attn.sif python3 train_peft.py
