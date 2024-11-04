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



