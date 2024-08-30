# LLM training Cookbook
 This repository provides scripts for fine-tuning a pre-trained language model (LLM) using the Hugging Face Transformers library. The scripts demonstrate how to load a pre-trained model, tokenize input text, and train the model on a custom dataset. The model is trained and tested on the question answering task using the [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) dataset. The performance could be compared to the Mistral-7B-Instruct-v0.2 models that was not trained from the cookbooks in LLM inference folder.

## Files
- **train_mistral_peft.py:** Script to fine-tune the Mistral-7B-Instruct-v0.2 model on the MedMCQA dataset.
- **mistral_trained_pred:** Script to run inference on the fine-tuned Mistral-7B-Instruct-v0.2 model on the MedMCQA dataset.
- **run_training.sh:** Shell script to execute `train_mistral_peft.py` on HPC Devana.
- **run_trained_inference.sh:** Shell script to execute `mistral_trained_pred.py` on HPC Devana.

---
**Note:** 
The `train_mistral_peft.py` script downloads data and models from the internet. Therefore, it is necessary to use a node with access to the internet (login node) to run this script for the first time.
The training does not fine-tune the whole model, but only tune the adapters using [PEFT](https://github.com/huggingface/peft). It is more efficient than fine-tuning the whole model. The saved adapters have less size than the model itself. The training script saves the model adapters in the data folder. The `mistral_trained_pred.py` script loads the adapters and you can specifi the checkpoint to load it from.




