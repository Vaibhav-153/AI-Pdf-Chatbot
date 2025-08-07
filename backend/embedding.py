# pdf_chatbot/backend/embedding.py

"""
Module for creating an advanced RAG retrieval system.

This module builds a sophisticated retriever that goes beyond simple semantic search.
It implements a hybrid approach by combining keyword-based search (BM25) with
semantic search (FAISS) and adds a re-ranking step for maximum relevance.

Key Improvements:
1.  Hybrid Search: Uses an EnsembleRetriever to combine the strengths of
    keyword (BM25) and semantic (FAISS) search.
2.  Advanced Re-ranking: Integrates a CohereRerank model to re-order
    retrieved documents based on contextual relevance to the query.
3.  Updated Embedding Model: Uses Google's newer 'text-embedding-latest'.
4.  In-Memory Operation: Designed to create the retriever in-memory for use
    within a user's session, which is faster and safer for concurrent users.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereRerank
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import (
    BM25Retriever,
    EnsembleRetriever,
    ContextualCompressionRetriever,
)
def create_hybrid_retriever(
    docs: List[Document],
    google_api_key: str,
    cohere_api_key: str,
    k: int = 10,
    rerank_top_n: int = 4
):
    """
    Creates an advanced hybrid retriever with a re-ranking step.

    This function sets up a multi-faceted retrieval system:
    1.  A keyword-based retriever (BM25) for exact matches.
    2.  A semantic vector-based retriever (FAISS) for contextual meaning.
    3.  An EnsembleRetriever to combine their results.
    4.  A CohereRerank model to intelligently re-order the combined results
        for the highest relevance.

    Args:
        docs: A list of LangChain Document objects to index.
        google_api_key: The API key for Google Generative AI (for embeddings).
        cohere_api_key: The API key for Cohere (for re-ranking).
        k: The total number of documents to retrieve initially from the ensemble.
        rerank_top_n: The number of top documents to return after re-ranking.

    Returns:
        A LangChain retriever object configured for hybrid search and re-ranking.
    """
    if not docs:
        raise ValueError("Cannot create retriever from an empty list of documents.")
    if not google_api_key or not cohere_api_key:
        raise ValueError("Google and Cohere API keys must be provided.")

    # 1. Initialize the embedding model
   # In backend/embedding.py
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # <-- This is the correct, stable model
        google_api_key=google_api_key
    )

    # 2. Create the two base retrievers: keyword and semantic
    
    # Keyword-based retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k # Number of docs to retrieve

    # Semantic vector-based retriever
    vectorstore = FAISS.from_documents(docs, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 3. Create the EnsembleRetriever to combine their strengths
    # The ensemble retriever will fetch documents from both sources. The weights
    # determine the importance of each retriever's results. 0.5/0.5 is a good start.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

   # 4. Create the document compressor (the re-ranker)
    compressor = CohereRerank(
        cohere_api_key=cohere_api_key,
        model="rerank-english-v3.0",
        top_n=rerank_top_n
    )

# 5. Create the final compression retriever
# This wraps the base retriever and applies the re-ranking compressor
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    print("Hybrid retriever with re-ranking created successfully.")
    return compression_retriever