# Named Entity Recognition (NER) with Transformers

This repository contains scripts and utilities for training DistilBERT model for Named Entity Recognition (NER). The model is fine-tuned on the CoNLL-2003 dataset, which includes named entities such as persons (PER), organizations (ORG), locations (LOC), and miscellaneous entities (MISC).

## Model Fine-tuning Details

- **Model:** DistilBERT (Base, Uncased)
- **Training Parameters:**
  - Learning Rate: 2e-5
  - Batch Size: 16
  - Number of Epochs: 5
  - Evaluation Metric: Loss

## Files and Directories

- **utils.py:** Utility functions including metrics computation and tokenization function.
- **train.py:** Script for fine-tuning DistilBERT for NER, includes evaluation on test set and saved training and validation history.
- **train_with_optuna.py:** Variant of train.py script that includes search for optimal set of hyperparameters using Optuna library.
- **inference.py:** Script for making predictions using the trained model.
- **run_train.sh:** Script to execute train.py on a HPC Devana using the specified container.
- **run_inference.sh:** Script to execute inference.py on a HPC Devana using the specified container.

---

**Note:** Customize paths and file names as necessary based on your project structure and requirements. The tutorial scripts download data and models from the internet, so it's necessary to use a node with internet access (login node) to run these scripts for the first time. The downloaded files will then be saved in cache for subsequent runs.
