# ==============================================
# FRONTEND WITH STREAMLIT: Academic Search App
# ==============================================

"""
Summary:
This Streamlit-based frontend enables users to query a backend academic document retrieval system (`rag`)
to find and summarize relevant research articles. Users can define a query, filter by category, specify how
many documents to retrieve, and reset the search. The app also provides language translation for abstracts
and displays metadata, similarity scores, and a synthesized assistant response.
"""

# Import necessary libraries
import os
import json
import torch
import streamlit as st
import _3_RAG_backend as rag  # Custom backend for retrieval-augmented generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configure the Streamlit page
st.set_page_config(page_title="Academic Research Assistance System", page_icon="📚")

# Initialize session state variables if they don't exist
if "query" not in st.session_state:
    st.session_state.query = ""  # User's research query
if "results" not in st.session_state:
    st.session_state.results = None  # Retrieved search results
if "run" not in st.session_state:
    st.session_state.run = False  # Controls search execution
if "just_reset" not in st.session_state:
    st.session_state.just_reset = False  # Prevents re-triggering after reset

# App title
st.title("📚 Academic Research Assistance System")

# Slider for selecting number of documents to retrieve
num_docs = st.slider(
    "Number of documents to retrieve:",
    min_value=1,
    max_value=10,
    value=3,
    key="num_docs"
)

# Load article categories from local JSON file
categories_path = os.path.join('data', 'processed', "arxiv_categories.json")
with open(categories_path, "r", encoding="utf-8") as json_file:
    categories = ['All categories'] + list(json.load(json_file).values())

# Dropdown for selecting a category filter
selected_category = st.selectbox(
    "Choose a category to filter the articles:",
    options=categories,
    index=0,
    key="category"
)


# Callback to set run flag when user inputs a new query
def set_run_flag():
    if not st.session_state.just_reset:
        st.session_state.run = True
    st.session_state.just_reset = False


# Text input for research query; triggers search on change
st.text_input("Introduce your research direction:", key="query", on_change=set_run_flag)

# Layout: two columns for Search and Reset buttons
col1, col2 = st.columns(2)

# Search button functionality
with col1:
    if st.button("🔎 Search for relevant articles"):
        if not st.session_state.query.strip():
            st.warning("Please, introduce a valid query.")
        else:
            with st.spinner("🔎 Searching articles..."):
                st.session_state.results = rag.search_and_answer(
                    st.session_state.query,
                    st.session_state.num_docs,
                    st.session_state.category
                )

# Reset button functionality
with col2:
    if st.button("🔄 Reset"):
        if st.session_state.results is None:
            st.info("No query to clear. Please enter one first.")
        else:
            # Clear session variables and show success message
            st.session_state.clear()
            st.session_state.run = False
            st.session_state.query = ""
            st.session_state.results = None
            st.session_state.just_reset = True
            st.success("The query has been cleared successfully.")

# Trigger automatic search if run flag is set and no results yet
if st.session_state.run and st.session_state.results is None:
    if not st.session_state.query.strip():
        st.warning("Please, introduce a valid query.")
    else:
        with st.spinner("🔎 Searching articles..."):
            st.session_state.results = rag.search_and_answer(
                st.session_state.query,
                st.session_state.num_docs,
                st.session_state.category
            )
    st.session_state.run = False

# Display results if available
if st.session_state.results:
    if st.session_state.results['answer'] is None:
        # Unsupported input language
        st.subheader("🌎 No language supported.")
        languages_path = os.path.join('data', 'processed', "language_dict.json")
        with open(languages_path, "r", encoding="utf-8") as json_file:
            supported_languages = ", ".join(list(json.load(json_file).values()) + ['english'])
        st.error(f"I’m sorry, this system just supports these languages: {supported_languages}")
    else:
        if st.session_state.results['num_retrieved'] == 0:
            # No documents found
            st.subheader("❌ No documents found.")
            st.error(
                "I’m sorry, but I do not have any additional specific articles in the current corpus on this topic. "
                "This does not imply that none exist. You may try expanding your search in other databases."
            )
        else:
            # Documents found: show count and query
            st.info(
                f"Found {st.session_state.results['num_retrieved']} relevant documents "
                f"(out of {st.session_state.num_docs} requested)."
            )
            st.subheader("🧠 Research direction:")
            st.markdown(f"> *{st.session_state.query}*")

            # Load translation model for abstract translation
            if st.session_state.results['language'] != 'en':
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                tokenizer_translator_docs = AutoTokenizer.from_pretrained(
                    f"Helsinki-NLP/opus-mt-en-{st.session_state.results['language']}"
                )
                translator_docs = AutoModelForSeq2SeqLM.from_pretrained(
                    f"Helsinki-NLP/opus-mt-en-{st.session_state.results['language']}"
                ).to(device)

            # Display each retrieved document
            st.subheader("📚 Found articles:")
            for idx, (doc, meta, dist) in enumerate(st.session_state.results['filtered_results'], start=1):
                with st.expander(f"📄 {idx}. {meta.get('title', 'Untitled')}"):
                    # Show DOI link if available
                    raw_doi = meta.get("doi", "").strip().rstrip(".")
                    if raw_doi:
                        doi_url = f"https://doi.org/{raw_doi}"
                        st.markdown(f"**DOI:** [{raw_doi}]({doi_url})")
                    else:
                        st.markdown("**DOI:** Not available")

                    # Show authors
                    st.markdown(f"**Authors:** {meta.get('authors', 'Not available')}")

                    # Translate and show abstract
                    if st.session_state.results['language'] != 'en':
                        batch = tokenizer_translator_docs([doc], return_tensors="pt").to(device)
                        generated_ids = translator_docs.generate(**batch)
                        doc = tokenizer_translator_docs.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    st.markdown(f"**Abstract:** {doc}")

                    # Show category and similarity metric
                    st.markdown(f"**Category:** {meta.get('category', 'Not available')}")
                    st.markdown(f"**Similarity metric (cosine distance):** {dist:.4f}")

            # Show assistant's answer
            st.subheader("📝 Assistant's answer:")
            st.write(f"{st.session_state.results['answer']}")

# streamlit run C:\Users\pablo\Desktop\NLP\RAG\_4_RAG_frontend.py
