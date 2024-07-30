# HPC-FineTuning-Examples
HPC Fine-Tuning Examples repository dedicated to providing examples and tutorials for fine-tuning language models using HPC system Devana.
Our goal is to help researchers and practitioners efficiently leverage HPC infrastructure to enhance their NLP workflows.

## Contents
- **Sentiment Analysis/**: This repository provides scripts to train a BERT model for sentiment analysis using the Twitter dataset. It includes data preprocessing, model training, evaluation, and inference.
- **Embedding Indices/**: This repository provides scripts for building information retrieval tool using BGE embeddings - BGE-M3 model, to retrieve similar documents from the 20 Newsgroups dataset using an Annoy index.

**Note:** Customize paths and file names as necessary based on your project structure and requirements.
The tutorial scripts download data and models from the internet, so it's necessary to use a node with internet access (login node) to run these scripts for the first time. The downloaded files will then be saved in cache for subsequent runs.

## Requirements

To use the examples provided in this repository, you will need the following:

- **Singularity**: Singularity-CE version 3.10 or higher is required. 
- **Recipe File**: A Singularity recipe file named `pt_devel_llm.recipe`.

There is an existing Singularity image named pt-2.3_llm.sif, built according to this recipe. It contains all necessary libraries, and the tutorials were built and tested on it.