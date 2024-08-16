import os
from sklearn.datasets import fetch_20newsgroups
from unidecode import unidecode
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from annoy import AnnoyIndex
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

# Function to preprocess input text data
def preprocess_text(text, stopwords):
    text = unidecode(text).lower() # turn to lowercase
    words = text.split() # split text into words
    filtered_words = [word for word in words if word not in stopwords] # remove stopwords
    return ' '.join(filtered_words)

# Fetch the dataset
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')#7532

print("Original document:")
print(newsgroups_train.data[0])

# Preprocess the dataset
newsgroups_train.data = [preprocess_text(doc, ENGLISH_STOP_WORDS) for doc in newsgroups_train.data]
newsgroups_test.data = [preprocess_text(doc, ENGLISH_STOP_WORDS) for doc in newsgroups_test.data]

# Display the first preprocessed document
print("\nPreprocessed document:")
print(newsgroups_train.data[0])

# Load BGE-M3 embedding model
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Function to create embeddings using the BGE-M3 model
def bge_m3_embed(queries, model_name):
    embeddings = model_name.encode(queries)['dense_vecs']
    return embeddings

# Create Annoy index
embedding_size = 1024  # Adjust based on actual embedding size from BGE-M3 model
annoy_index = AnnoyIndex(embedding_size, 'angular')

batch_size = 2 # Adjust batch size for efficiency
embeddings = []

# Batch processing and embedding creation
for i in tqdm(range(0, len(newsgroups_train.data), batch_size)):
    batch = newsgroups_train.data[i:i+batch_size]
    batch_embeddings = bge_m3_embed(batch, model)
    embeddings.extend(batch_embeddings)

# Adding items to Annoy index
for i, embedding in enumerate(embeddings):
    annoy_index.add_item(i, embedding)    

# Build and save the Annoy index
annoy_index.build(10)  # n of trees
annoy_index.save('newsgroups.ann')
print("Annoy index created and saved successfully.")


