# pdf_chatbot/backend/embedding.py

"""
Module for handling vector embeddings and the vector store.

This script is responsible for the core of Rule #2 (Semantic RAG Search).
It takes processed text chunks, uses a Google embedding model to convert them
into high-dimensional vectors, and stores them in a FAISS vector store for
fast and efficient similarity searches.
"""

import os
from typing import List
from langchain_core.documents import Document

# Key LangChain components for vector store operations
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Define a constant for the local storage path of the vector store index.
# This makes it easy to reference and manage.
VECTOR_STORE_PATH = "faiss_index"

def get_vector_store(text_chunks: List[Document], api_key: str) -> None:
    """
    Creates and saves a FAISS vector store from document chunks.

    This function will overwrite any existing vector store. It processes the
    chunks, generates embeddings via the Google API, and saves the resulting
    index to a local directory defined by VECTOR_STORE_PATH.

    Args:
        text_chunks: A list of text chunks (Document objects) from ingestion.
        api_key: The Google API key for authenticating with the embedding service.
    
    Returns:
        None. The function saves the index to disk as a side effect.
    """

    if not text_chunks:
        print("Warning: Received empty list of text chunks. No vector store will be created.")
        return

    try:
        # Initialize the embedding model. "models/embedding-001" is a powerful and
        # cost-effective choice for generating high-quality embeddings.
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

        # Create the FAISS vector store from the chunks and embedding model.
        # This is the most computationally intensive step in the offline process.
        print("Generating embeddings and creating FAISS vector store...")
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Save the index to a local path for persistence. This allows the app
        # to load the knowledge base instantly on subsequent runs without reprocessing.
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"Vector store created and saved successfully at: {VECTOR_STORE_PATH}")

    except Exception as e:
        print(f"An error occurred during vector store creation: {e}")
        # This can be critical, so re-raising might be an option in a production system.
        raise e


def load_vector_store(api_key: str) -> FAISS:
    """
    Loads an existing FAISS vector store from the local disk.

    This function is used during the query phase to quickly load the pre-built
    knowledge base into memory for searching.

    Args:
        api_key: The Google API key, required to initialize the same embedding
                 model used during the creation of the index.

    Returns:
        The loaded FAISS vector store object, or None if it cannot be found.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"Vector store not found at '{VECTOR_STORE_PATH}'. Please process documents first.")
        return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        # The 'allow_dangerous_deserialization' flag is required by LangChain's
        # new security model when loading local FAISS indexes.
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
        return vector_store

    except Exception as e:
        print(f"An error occurred while loading the vector store: {e}")
        return None