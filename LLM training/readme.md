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

## Hugging Face Token Setup
Some of the scripts require that you have account and the [access token](https://huggingface.co/docs/hub/en/security-tokens) generated at the [Hugging Face](https://huggingface.co/) website. The token is used to access some models. The token should be added to the code as a string as shown at the [access token](https://huggingface.co/docs/hub/en/security-tokens).

### Example: Loading a Model with an Access Token

To load a model using an access token, you can use the following code snippet:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "your-model-id"
access_token = "your-access-token"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=access_token,  # Add the access token here
    device_map='cuda'
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=access_token  # Add the access token here
)
```

Before accessing certain models, you may need to agree to the conditions on the model card on the Hugging Face website, including sharing your contact information (email and username) with the repository authors.

## LLM Training Benchmark

### [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) single GPU inference on [Devana HPC](https://userdocs.nscc.sk/devana/system_overview/introduction/) 

Evaluate [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) without, with 8-bit and with 4-bit [bitsandbytes quantisation](https://huggingface.co/docs/bitsandbytes/index) on the [MedMCQA](https://medmcqa.github.io) dataset. 

| Setup | GPUs |  Inference speed | Inference GPU memory |
| - | - | - | - |
| w/o bnb | 1 | 2.3 samples/s | 16.6 GB |
| w/o bnb | 2 | 2.4 samples/s | 16.6 + 0.6 GB |
| w/o bnb | 3 | 2.4 samples/s | 16.6 + 0.6 GB |
| w/o bnb | 4 | 2.4 samples/s | 16.6 + 0.6 GB |
| w/ bnb (8bit) | 1 | 7.9 samples/s | 19.8 samples/s |
| w/ bnb (4bit) | 1 | 1.3 samples/s | 7.3 GB |


