# ======================================================
# EMBEDDING AND INDEXING SCRIPT: Populate Vector DB
# ======================================================

"""
Summary:
This script embeds academic documents using the `nomic-embed-text` model via Ollama,
and stores the embeddings into a persistent ChromaDB vector database.
It avoids re-processing already indexed documents by checking existing IDs, processes
data in batches, and stores embeddings along with metadata for efficient semantic search.
"""

# Import necessary libraries
import os
import chromadb
import ollama
import pandas as pd
from tqdm import tqdm  # For progress visualization

# Initialize a persistent ChromaDB client (creates or connects to local DB)
client = chromadb.PersistentClient(path="./vector_database")

# Retrieve or create a named collection for storing document embeddings
collection = client.get_or_create_collection(name="docs")

# Get list of already processed document IDs to avoid duplication
processed = collection.get()['ids']

# Load cleaned dataset
data_path = os.path.join('data', 'processed', 'data_nlp_cleaned.csv')
data = pd.read_csv(data_path, dtype={0: str})

# Filter out already indexed documents based on IDs
data = data[~data['id'].isin(processed)]

# Set batch size for efficient processing
batch_size = 32

# Calculate number of batches for progress bar
num_batches = (len(data) + batch_size - 1) // batch_size

# Process data in batches
for i in tqdm(range(0, len(data), batch_size), total=num_batches, desc="Processing batches"):
    batch = data.iloc[i:i+batch_size]  # Slice batch

    # Extract contents to embed
    contents = batch['content'].tolist()

    # Generate embeddings using Ollama with specified model
    embeddings = ollama.embed(model="nomic-embed-text", input=contents)["embeddings"]

    # Extract additional fields for indexing
    ids = batch['id'].astype(str).tolist()
    documents = batch['abstract'].tolist()
    metadatas = batch[['category', 'authors', 'doi', 'title']].to_dict(orient='records')

    # Add embeddings and metadata to ChromaDB collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
