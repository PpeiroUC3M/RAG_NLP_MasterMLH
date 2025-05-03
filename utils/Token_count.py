# =======================================================
# TOKEN COUNT VALIDATOR: Detect Over-Limit Documents
# =======================================================

"""
Summary:
This script reads a list of cleaned academic documents and uses the `tiktoken` tokenizer
to count the number of tokens in each one. It flags and prints any documents that exceed
a predefined token limit (e.g., for compatibility with language models like GPT-4).
"""

# Import necessary libraries
import os
import tiktoken
import pandas as pd


# ====================================
# FUNCTION TO COUNT TOKENS IN TEXT
# ====================================

def count_tokens(text: str) -> int:
    """
    Tokenizes the input text using the selected encoding and returns the token count.
    """
    tokens = tokenizer.encode(text)
    return len(tokens)


# ========================
# LOAD AND PROCESS DOCUMENTS
# ========================

# Load 'content' column from cleaned CSV as a list of documents
docs_path = os.path.join('data', 'processed', 'data_nlp_cleaned.csv')
documents = pd.read_csv(docs_path, dtype={0: str})['content'].tolist()

# Initialize tokenizer with OpenAI's cl100k_base encoding (used for GPT-4, GPT-3.5-turbo)
tokenizer = tiktoken.get_encoding("cl100k_base")

# Define token limit per document (e.g., for GPT-4 context window)
TOKEN_LIMIT = 8192

# Iterate over each document, count tokens, and flag if over the limit
for i, doc in enumerate(documents):
    num_tokens = count_tokens(doc)
    if num_tokens > TOKEN_LIMIT:
        print(f"🔴 Document {i} exceeds the limit: {num_tokens} tokens")
