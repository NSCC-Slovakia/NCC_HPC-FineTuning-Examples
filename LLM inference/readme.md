# LLM inference Cookbook
This repository contains scripts for running inference on a pre-trained language model (LLM) using the Hugging Face Transformers library. The scripts demonstrate how to load a pre-trained model, tokenize input text, and generate output text. The repository includes a script for running inference on the AYA-101 and Mistral-7B-Instruct-v0.2 models. The models are tested on the question answering task using the [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) dataset.

## Files
- **aya_pred.py:** Script to run inference on the AYA-101 model using the MedMCQA dataset.
- **mistral_pred.py:** Script to run inference on the Mistral-7B-Instruct-v0.2 model using the MedMCQA dataset.
- **run_inference_aya.sh:** Shell script to execute `aya_pred.py` on HPC Devana.
- **run_inference_mistral.sh:** Shell script to execute `mistral_pred.py` on HPC Devana.

---

**Note:** 
The both script downloads data and models from the internet. Therefore, it is necessary to use a node with access to the internet (login node) to run this script for the first time.




