import os
from sklearn.datasets import fetch_20newsgroups
from unidecode import unidecode
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from annoy import AnnoyIndex
from FlagEmbedding import BGEM3FlagModel
import argparse

# Function to preprocess input text data
def preprocess_text(text, stopwords):
    text = unidecode(text).lower() # turn to lowercase
    words = text.split() # split text into words
    filtered_words = [word for word in words if word not in stopwords] # remove stopwords
    return ' '.join(filtered_words) 

# Function to create embeddings using the BGE-M3 model
def bge_m3_embed(queries, model_name):
    embeddings = model_name.encode(queries)['dense_vecs']
    return embeddings

def main(args):
    # Load the Annoy index
    embedding_size = 1024
    annoy_index = AnnoyIndex(embedding_size, 'angular') 
    annoy_index.load('newsgroups.ann') # load created annoy vector database

    # Fetch the dataset to get original documents
    newsgroups_train = fetch_20newsgroups(subset='train')

    # Load BGE-M3 embedding model
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    # Preprocess the query
    query_text_processed = preprocess_text(args.query, ENGLISH_STOP_WORDS)

    # Get query embedding
    query_embedding = bge_m3_embed(query_text_processed, model)

    # Search for top k most similar documents in the Annoy index
    similar_item_indices = annoy_index.get_nns_by_vector(query_embedding, args.n_neighbors, include_distances=True)

    # Print the results
    for idx, distance in zip(*similar_item_indices):
        print(f"Document index: {idx}, Distance: {distance}")
        print(f"Document: {newsgroups_train.data[idx]}")
        print(f"Doc target: {newsgroups_train.target_names[newsgroups_train.target[idx]]}")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve similar documents from 20 Newsgroups dataset using Annoy index.")
    parser.add_argument('--query', type=str, required=True, help='Query text to search for similar documents.')
    parser.add_argument('--n_neighbors', type=int, default=10, help='Number of nearest neighbors to retrieve.')

    args = parser.parse_args()
    main(args)