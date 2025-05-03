# =============================================
# DATA CLEANING SCRIPT: arXiv Metadata Cleaner
# =============================================

"""
Summary:
This script reads the raw arXiv metadata JSON file line-by-line, extracts key fields
(e.g., title, abstract, authors), cleans LaTeX and special formatting from the textual data,
and saves the cleaned result as CSV files. One full version is saved, and a second version
includes only the first 1,000 entries for trial or testing purposes.
"""

# Import necessary libraries
import os
import re
import json
import pandas as pd

# Define the path to the raw arXiv metadata JSON file
json_path = os.path.join('data', 'raw', 'arxiv-metadata-oai-snapshot.json')

# Initialize a list to store parsed paper entries
data = []

# Open and read the JSON file line-by-line (each line is a JSON object)
with open(json_path, 'r') as f:
    for i, line in enumerate(f):
        try:
            # Parse JSON line into dictionary
            paper = json.loads(line)
            # Extract selected fields with default fallbacks
            data.append({
                "id": paper.get("id"),
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "category": paper.get("categories"),
                "authors": paper.get("authors"),
                "doi": paper.get("doi"),
            })
        except json.JSONDecodeError:
            # Skip malformed JSON lines
            continue

# Convert to DataFrame, drop rows missing title or abstract, fill remaining NaNs
data = pd.DataFrame(data).dropna(subset=["title", "abstract"]).fillna("Not found")


# =======================
# TEXT CLEANING FUNCTION
# =======================

def clean_texts(text):
    """
    Cleans LaTeX markup and formatting artifacts from input text:
    - Removes comments, LaTeX environments (e.g. figure, table)
    - Replaces inline and display math with [FORMULA]
    - Strips special characters, backslashes, excess whitespace
    """
    text = re.sub(r'(?<!\\)%.*', '', text)  # Remove LaTeX comments
    text = re.sub(r'\\begin{(figure|table|equation|align|tikzpicture|lstlisting)}.*?\\end{\1}', '', text,
                  flags=re.DOTALL)  # Remove LaTeX environments
    text = re.sub(r'\\[a-zA-Z]+\*?{([^}]*)}', r'\1', text)  # Simplify LaTeX commands with content
    text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:{[^}]*})?', '', text)  # Remove remaining LaTeX commands
    text = re.sub(r'\$.*?\$', '[FORMULA]', text)  # Replace inline math
    text = re.sub(r'\\\((.*?)\\\)', '[FORMULA]', text)  # Replace \( \)
    text = re.sub(r'\\\[(.*?)\\\]', '[FORMULA]', text)  # Replace \[ \]
    text = re.sub(r'\\\\', ' ', text)  # Replace line breaks
    text = re.sub(r'\\[\'\"\`\^~=][a-zA-Z]', '', text)  # Remove accented characters
    text = re.sub(r'\\', '', text)  # Remove lone backslashes
    text = text.replace('\n', ' ').replace('\t', ' ')  # Remove newlines and tabs
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces

    return text.strip()


# Apply cleaning to each relevant text column
data['abstract'] = data['abstract'].apply(clean_texts)
data['title'] = data['title'].apply(clean_texts)
data['authors'] = data['authors'].apply(clean_texts)
data['category'] = data['category'].apply(clean_texts)

# Create a combined 'content' column for later use (e.g., embedding)
data['content'] = data["title"] + ". " + data["abstract"]

# =======================
# SAVE CLEANED DATASETS
# =======================

# Save full cleaned dataset
saving_path_cleaned = os.path.join('data', 'processed', 'data_nlp_cleaned.csv')
data.to_csv(saving_path_cleaned, index=False)

# Save a smaller trial dataset (first 1,000 rows)
saving_path_trials = os.path.join('data', 'processed', 'data_trials.csv')
data.iloc[:1000, :].to_csv(saving_path_trials, index=False)
