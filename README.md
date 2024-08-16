# HPC-FineTuning-Examples
HPC Fine-Tuning Examples repository dedicated to providing examples and tutorials for fine-tuning language models using HPC system Devana.
Our goal is to help researchers and practitioners efficiently leverage HPC infrastructure to enhance their NLP workflows.

## Contents
- **Embedding Indices**: This repository provides scripts for building information retrieval tool using BGE embeddings - BGE-M3 model, to retrieve similar documents from the 20 Newsgroups dataset using an Annoy index.
- **NER**: This repository contains scripts and utilities for training DistilBERT model for Named Entity Recognition (NER). The model is fine-tuned on the CoNLL-2003 dataset, which includes named entities such as persons (PER), organizations (ORG), locations (LOC), and miscellaneous entities (MISC).
![Highlighted Sentence](NER_image.png)
- **Sentiment Analysis**: This repository provides scripts to train a BERT model for sentiment analysis using the Twitter dataset. It includes data preprocessing, model training, evaluation, and inference.

**Note:** Customize paths and file names as necessary based on your project structure and requirements.
The tutorial scripts download data and models from the internet, so it's necessary to use a node with internet access (login node) to run these scripts for the first time. The downloaded files will then be saved in cache for subsequent runs.

## Requirements

To use the examples provided in this repository, you have two options:

1. **Use an Existing Singularity Image**: Utilize the existing Singularity image named `pt-2.3_llm.sif`, which contains all the necessary libraries.
2. **Create a New Singularity Image**: Follow the instructions below to create a new Singularity image.

### Instructions for Creating a New Singularity Image:

- **Singularity**: Ensure you have Singularity-CE version 3.10 or higher installed.
- **Recipe File**: Use the Singularity recipe file named `pt_devel_llm.recipe`.
- **Build Command**: To build a Singularity image from the recipe file, use the following command:

  ```sh
  sudo singularity build test.sif pt_devel_llm.recipe