import os
import json
import torch
import ollama
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
from langdetect import DetectorFactory

DetectorFactory.seed = 0

ground_truth_titles = {
    "multimodal sarcasm detection": [
        "Towards Multimodal Sarcasm Detection (An _Obviously_ Perfect Paper)",
        "Detecting Sarcasm in Multimodal Social Platforms",
        "FiLMing Multimodal Sarcasm Detection with Attention",
        "Multimodal Learning using Optimal Transport for Sarcasm and Humor Detection",
        "Debiasing Multimodal Sarcasm Detection with Contrastive Learning",
        "Computational Sarcasm Analysis on Social Media: A Systematic Review"
        "Multi-View Incongruity Learning for Multimodal Sarcasm Detection",
        "AMuSeD: An Attentive Deep Neural Network for Multimodal Sarcasm Detection Incorporating Bi-modal Data Augmentation",
        "RCLMuFN: Relational Context Learning and Multiplex Fusion Network for Multimodal Sarcasm Detection"
    ],
    "sentiment analysis in social media texts": [
        "Adaptation of domain-specific transformer models with text oversampling for sentiment analysis of social media posts on Covid-19 vaccines",
        "Emotion Detection and Analysis on Social Media",
        "Identifying negativity factors from social media text corpus using sentiment analysis method",
        "Sentiment Identification in Code-Mixed Social Media Text",
        "CMSAOne@Dravidian-CodeMix-FIRE2020: A Meta Embedding and Transformer model for Code-Mixed Sentiment Analysis on Social Media Text",
        "C1 at SemEval-2020 Task 9: SentiMix: Sentiment Analysis for Code-Mixed Social Media Text using Feature Engineering",
        "Sentiment Analysis on Social Media Content",
        "IUST at SemEval-2020 Task 9: Sentiment Analysis for Code-Mixed Social Media Text using Deep Neural Networks and Linear Baselines",
        "NITS-Hinglish-SentiMix at SemEval-2020 Task 9: Sentiment Analysis For Code-Mixed Social Media Text Using an Ensemble Model",
        "Social Media Analysis for Product Safety using Text Mining and Sentiment Analysis"
    ],
    "factual consistency in summarization": [
        "Factual consistency evaluation of summarization in the Era of large language models",
        "Improving Factual Consistency of News Summarization by Contrastive Preference Optimization",
        "Enhancing Factual Consistency of Abstractive Summarization",
        "Factually Consistent Summarization via Reinforcement Learning with Textual Entailment Feedback",
        "On Improving Summarization Factual Consistency from Natural Language Feedback",
        "Evaluating the Factual Consistency of Abstractive Text Summarization",
        "Asking and Answering Questions to Evaluate the Factual Consistency of Summaries",
        "Evaluating the Factual Consistency of Large Language Models Through News Summarization",
        "Entity-based SpanCopy for Abstractive Summarization to Improve the Factual Consistency",
        "Discourse Understanding and Factual Consistency in Abstractive Summarization",
        "CO2Sum:Contrastive Learning for Factual-Consistent Abstractive Summarization",
        "Entity-level Factual Consistency of Abstractive Text Summarization",
    ],
    "multilingual named entity recognition": [
        "FewTopNER: Integrating Few-Shot Learning with Topic Modeling and Named Entity Recognition in a Multilingual Framework",
        "MultiCoNER v2: a Large Multilingual dataset for Fine-grained and Noisy Named Entity Recognition",
        "Building Multilingual Corpora for a Complex Named Entity Recognition and Classification Hierarchy using Wikipedia and DBpedia",
        "Multilingual Name Entity Recognition and Intent Classification Employing Deep Learning Architectures",
        '"Translation can\'t change a name": Using Multilingual Data for Named Entity Recognition',
        "CMNEROne at SemEval-2022 Task 11: Code-Mixed Named Entity Recognition by leveraging multilingual data",
        "Multilingual Named Entity Recognition Using Pretrained Embeddings, Attention Mechanism and NCRF",
        "Sources of Transfer in Multilingual Named Entity Recognition",
        "Robust Multilingual Named Entity Recognition with Shallow Semi-Supervised Features",
        "Improving Multilingual Named Entity Recognition with Wikipedia Entity Type Mapping"
    ],
    "logical reasoning with transformers": [
        "Can Transformers Reason Logically? A Study in SAT Solving",
        "Assessing Logical Reasoning Capabilities of Encoder-Only Transformer Models",
        "Teaching Probabilistic Logical Reasoning to Transformers",
        "Transformers as Soft Reasoners over Language",
        "Logiformer: A Two-Branch Graph Transformer Network for Interpretable Logical Reasoning",
        "Join-Chain Network: A Logical Reasoning View of the Multi-head Attention in Transformer"
    ],
    "hierarchical document encoding": [
        "A Hierarchical Encoding-Decoding Scheme for Abstractive Multi-document Summarization",
        "Hierarchical Document Encoder for Parallel Corpus Mining",
        "Interpretable Structure-aware Document Encoders with Hierarchical Attention",
        "An Analysis of Hierarchical Text Classification Using Word Embeddings",
        "Language Model Pre-training for Hierarchical Document Representations"
    ],
    "bayesian optimization for hyperparameter tuning": [
        "Alleviating Hyperparameter-Tuning Burden in SVM Classifiers for Pulmonary Nodules Diagnosis with Multi-Task Bayesian Optimization",
        "Bayesian Optimization for Hyperparameters Tuning in Neural Networks",
        "Bayesian Optimization is Superior to Random Search for Machine Learning Hyperparameter Tuning: Analysis of the Black-Box Optimization Challenge 2020",
        "Efficient hyperparameter tuning for kernel ridge regression with Bayesian optimization",
        "Practical Multi-fidelity Bayesian Optimization for Hyperparameter Tuning",
        "Fast Hyperparameter Tuning using Bayesian Optimization with Directional Derivatives"
    ],
    "transfer learning question answering": [
        "Question Answering through Transfer Learning from Large Fine-grained Supervision Data",
        "Transfer Learning Enhanced Single-choice Decision for Multi-choice Question Answering",
        "Few-shot Transfer Learning for Knowledge Base Question Answering: Fusing Supervised Models with In-Context Learning",
        "DTW at Qur'an QA 2022: Utilising Transfer Learning with Transformers for Question Answering in a Low-resource Domain",
        "Cross-Lingual Transfer Learning for Question Answering",
        "Supervised Transfer Learning for Product Information Question Answering",
        "Transfer Learning via Unsupervised Task Discovery for Visual Question Answering",
        "Supervised and Unsupervised Transfer Learning for Question Answering",
        "Question Answering through Transfer Learning from Large Fine-grained Supervision Data",
    ],
    "deep learning for medical image processing": [
        "Evaluation of Extra Pixel Interpolation with Mask Processing for Medical Image Segmentation with Deep Learning",
        "A Gentle Introduction to Deep Learning in Medical Image Processing",
        "Deep Learning for Medical Image Processing: Overview, Challenges and Future"
    ],
    "prototype-based neural networks": [
        "A Prototype-Based Neural Network for Image Anomaly Detection and Localization",
        "ProtoGate: Prototype-based Neural Networks with Global-to-local Feature Selection for Tabular Biomedical Data",
        "PAGE: Prototype-Based Model-Level Explanations for Graph Neural Networks",
        "Towards Prototype-Based Self-Explainable Graph Neural Network",
        "Towards Deep Machine Reasoning: a Prototype-based Deep Neural Network with Decision Tree Inference",
        "Prototype-based Neural Network Layers: Incorporating Vector Quantization",
    ]
}

client = chromadb.PersistentClient(path="./vector_database")
collection = client.get_or_create_collection(name="docs")

languages_path = os.path.join('data', 'processed', "language_dict.json")
with open(languages_path, "r", encoding="utf-8") as json_file:
    supported_languages = list(json.load(json_file).keys()) + ['en']  # Add English manually


def search_and_answer(query: str, num_docs: int = 10, category: str = 'All categories'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_lang = detect(query)
    vector_database_lang = 'en'

    if query_lang in supported_languages:

        if query_lang == 'en':
            query_in_english = query
        else:
            tokenizer_translator_query = AutoTokenizer.from_pretrained(
                f"Helsinki-NLP/opus-mt-{query_lang}-{vector_database_lang}")
            translator_query = AutoModelForSeq2SeqLM.from_pretrained(
                f"Helsinki-NLP/opus-mt-{query_lang}-{vector_database_lang}").to(device)

            batch = tokenizer_translator_query([query], return_tensors="pt").to(device)
            generated_ids = translator_query.generate(**batch)
            query_in_english = tokenizer_translator_query.batch_decode(generated_ids, skip_special_tokens=True)[0]

        response = ollama.embed(model="nomic-embed-text", input=query_in_english)["embeddings"]

        results = collection.query(
            query_embeddings=response,
            n_results=num_docs,
            include=['documents', 'metadatas', 'distances'],
        )

        similarity_threshold = 0.7

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if category == 'All categories':
            filtered_results = [
                (doc, meta, dist)
                for doc, meta, dist in zip(documents, metadatas, distances)
                if dist < similarity_threshold
            ]
        else:
            categories_path = os.path.join('data', 'processed', "arxiv_categories.json")
            with open(categories_path, "r", encoding="utf-8") as json_file:
                category = [k for k, v in json.load(json_file).items() if v == category][0]

            filtered_results = [
                (doc, meta, dist)
                for doc, meta, dist in zip(documents, metadatas, distances)
                if dist < similarity_threshold and category in meta['category'].split()
            ]

        retrieved_important = []
        retrieved_titles = [meta.get("title", "").strip().rstrip(".") for _, meta, _ in filtered_results]
        relevant_found = ground_truth_titles[query]
        for title in retrieved_titles:
            if title in relevant_found:
                retrieved_important.append(title)
        if len(retrieved_important) > 0:
            return 1
        else:
            return 0

    return None


recalls = [1, 3, 5, 10]
recalls_results = []
for i in recalls:
    found = 0
    searches = 0
    for j in ground_truth_titles.keys():
        found += search_and_answer(j, num_docs=i)
        searches += 1
    recalls_results.append(found/searches)

print(recalls_results)