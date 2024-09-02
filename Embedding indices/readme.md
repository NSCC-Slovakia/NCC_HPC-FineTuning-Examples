# BGE Embeddings and Retrieval Cookbook
This repository provides scripts to build BGE embeddings using the BGE-M3 model and retrieve similar documents from the 20 Newsgroups dataset using an Annoy index.

## Files
- **bge_embeddings.py:** Script to preprocess the 20 Newsgroups dataset, generate BGE embeddings using the BGE-M3 model, and build an Annoy index for efficient retrieval.
- **bge_retrieval.py:** Script to retrieve similar documents from the 20 Newsgroups dataset based on a query using the pre-built Annoy index and BGE-M3 embeddings.
- **run_embeddings.sh:** Shell script to execute bge_embeddings.py on HPC Devana.
- **run_retrieval.sh:** Shell script to execute bge_retrieval.py on HPC Devana. Specify your query and number of files to retrieve.

---

**Note:** Customize paths and file names as necessary based on your project structure and requirements.
The `bge_embeddings.py` script downloads data and models from the internet. Therefore, it is necessary to use a node with access to the internet to run this script for the first time.




