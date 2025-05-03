# ===================================================
# CATEGORY SCRAPER: Extract arXiv Category Metadata
# ===================================================

"""
Summary:
This script scrapes the official arXiv category taxonomy page to extract all available
categories and their descriptions. It processes the HTML structure to find category codes
and names, and saves them into a structured JSON file for downstream use.
"""

# Import required libraries
import os
import json
import requests
from bs4 import BeautifulSoup


# ===================================
# FUNCTION TO SCRAPE arXiv CATEGORIES
# ===================================

def get_arxiv_categories():
    """
    Fetches the arXiv category taxonomy page, parses it, and extracts category codes
    and human-readable names from <h4> tags. Returns a dictionary mapping codes to names.
    """
    url = "https://arxiv.org/category_taxonomy"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    categories = {}

    # Extract category info from <h4> tags that include both code and name
    for h4 in soup.find_all("h4"):
        text = h4.get_text(" ", strip=True)
        if "(" in text and ")" in text:
            parts = text.split(" (")
            if len(parts) == 2:
                code = parts[0].strip()            # e.g., "cs.AI"
                name = parts[1].strip(") ")         # e.g., "Artificial Intelligence"
                categories[code] = name

    return categories


# =========================
# EXECUTION ENTRY POINT
# =========================

if __name__ == "__main__":
    # Fetch and parse arXiv categories
    arxiv_categories = get_arxiv_categories()

    # Remove the first key, which is often a general heading or invalid entry
    first_key = next(iter(arxiv_categories))
    del arxiv_categories[first_key]

    # Define output path and save categories as JSON
    saving_path = os.path.join('data', 'processed', "arxiv_categories.json")
    with open(saving_path, "w", encoding="utf-8") as json_file:
        json.dump(arxiv_categories, json_file, ensure_ascii=False, indent=4)
