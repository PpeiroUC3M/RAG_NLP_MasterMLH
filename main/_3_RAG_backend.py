# ========================================================
# BACKEND: SEARCH & ANSWER WITH TRANSLATION + RAG SYSTEM
# ========================================================

"""
Summary:
This script defines the core backend logic for a multilingual academic research assistant.
It detects the language of the input query, translates it to English if needed, retrieves
relevant documents from a ChromaDB vector database using semantic embeddings (via Ollama),
filters by similarity and category, and constructs a structured answer with LLM-based reasoning.
If the query is not in a supported language, it returns None.
"""

# ================
# Import libraries
# ================
import os
import json
import torch
import ollama
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
from langdetect import DetectorFactory

# Fix seed for consistent language detection
DetectorFactory.seed = 0

# ========================================
# Initialize ChromaDB vector DB client
# ========================================
client = chromadb.PersistentClient(path="./vector_database")
collection = client.get_or_create_collection(name="docs")

# ===============================
# Load supported language list
# ===============================
languages_path = os.path.join('data', 'processed', "language_dict.json")
with open(languages_path, "r", encoding="utf-8") as json_file:
    supported_languages = list(json.load(json_file).keys()) + ['en']  # Add English manually


# =====================================================
# Function: search_and_answer
# Main logic for processing user queries
# =====================================================
def search_and_answer(query: str, num_docs: int = 10, category: str = 'All categories'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Detect query language
    query_lang = detect(query)
    vector_database_lang = 'en'

    # Proceed only if the query is in a supported language
    if query_lang in supported_languages:

        # Translate query if necessary
        if query_lang == 'en':
            query_in_english = query
        else:
            # Load tokenizer and model for query → English translation
            tokenizer_translator_query = AutoTokenizer.from_pretrained(
                f"Helsinki-NLP/opus-mt-{query_lang}-{vector_database_lang}")
            translator_query = AutoModelForSeq2SeqLM.from_pretrained(
                f"Helsinki-NLP/opus-mt-{query_lang}-{vector_database_lang}").to(device)

            # Generate translation
            batch = tokenizer_translator_query([query], return_tensors="pt").to(device)
            generated_ids = translator_query.generate(**batch)
            query_in_english = tokenizer_translator_query.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Generate embedding for the query using Ollama
        response = ollama.embed(model="nomic-embed-text", input=query_in_english)["embeddings"]

        # Query vector database for similar documents
        results = collection.query(
            query_embeddings=response,
            n_results=num_docs,
            include=['documents', 'metadatas', 'distances'],
        )

        # Set similarity threshold (lower distance = more similar)
        similarity_threshold = 0.7

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Filter results based on threshold and optional category
        if category == 'All categories':
            filtered_results = [
                (doc, meta, dist)
                for doc, meta, dist in zip(documents, metadatas, distances)
                if dist < similarity_threshold
            ]
        else:
            # Load arXiv category codes to match human-readable input
            categories_path = os.path.join('data', 'processed', "arxiv_categories.json")
            with open(categories_path, "r", encoding="utf-8") as json_file:
                category = [k for k, v in json.load(json_file).items() if v == category][0]

            filtered_results = [
                (doc, meta, dist)
                for doc, meta, dist in zip(documents, metadatas, distances)
                if dist < similarity_threshold and category in meta['category'].split()
            ]

        # If no relevant documents were found
        if not filtered_results:
            response = (
                f"No relevant documents were found for the query: {query}. "
                f"It is not possible to perform the requested comparison."
            )
        else:
            # Construct the context from filtered documents
            contexto = "\n\n".join([doc for doc, _, _ in filtered_results])

            # Build prompt for LLM-based summarization and analysis
            prompt = f"""
            You are an academic research assistant.

            Your task is to help a researcher analyze the state of the art. Use only the information found in the 
            provided documents. If you cannot find enough relevant information, simply state so.

            The answer must contain clearly defined the following three points:
            1. Briefly summarize the existing works related to the topic.
            2. Compare the user's proposed research direction with the existing works.
            3. Highlight the main differences and potential contributions of the new direction.

            Once you respond to these prompts, end the conversation without interacting further with the user.

            ---
            Retrieved documents:
            {contexto}

            ---
            Research direction (user's question):
            {query}

            ---
            Response:
            """

            # Send prompt to LLM via Ollama
            response = ollama.chat(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.2,
                    "top_p": 0.85,
                    "repeat_penalty": 0.8,
                }
            )
            response = response['message']['content']

            # Translate response back to original query language if needed
            if query_lang != 'en':
                tokenizer_translator_response = AutoTokenizer.from_pretrained(
                    f"Helsinki-NLP/opus-mt-{vector_database_lang}-{query_lang}")
                translator_response = AutoModelForSeq2SeqLM.from_pretrained(
                    f"Helsinki-NLP/opus-mt-{vector_database_lang}-{query_lang}").to(device)

                # Split response into chunks and translate line-by-line
                text_chunks = response.split('\n')
                translated_chunks = []

                for chunk in text_chunks:
                    if chunk.strip() == "":
                        translated_chunks.append("")
                        continue
                    batch = tokenizer_translator_response([chunk], return_tensors="pt").to(device)
                    generated_ids = translator_response.generate(**batch)
                    translated = tokenizer_translator_response.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    translated_chunks.append(translated)

                # Reconstruct translated response
                response = "\n".join(translated_chunks)

    else:
        # Query language not supported
        filtered_results = []
        response = None

    # Return results, final response, and detected language
    return {
        "filtered_results": filtered_results,
        "answer": response,
        "num_retrieved": len(filtered_results),
        "language": query_lang
    }
