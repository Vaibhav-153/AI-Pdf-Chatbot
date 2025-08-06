# pdf_chatbot/backend/ingestion.py

"""
Module for data ingestion from PDF documents.

Handles the initial step of the RAG pipeline: extracting raw text and
metadata from uploaded PDF files. This is critical for Rule #2 (Semantic Search)
and Rule #3 (Multi-PDF Handling), as it preserves the source of every
piece of information for later citation.
"""

import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Union
from langchain_core.documents import Document

def get_pdf_text_and_metadata(pdf_docs: List[Union[str, any]]) -> List[Document]:
    """
    Extracts text and structured metadata from a list of PDF files.

    Each page of each PDF is treated as a separate Document object, containing
    the page's content and a metadata dictionary with the source filename and
    page number.

    Args:
        pdf_docs: A list of uploaded file objects (from Streamlit) or file paths.

    Returns:
        A list of LangChain Document objects, one for each page of the PDFs.
    """
    documents = []
    for pdf in pdf_docs:
        if pdf is None:
            continue
        
        is_file_path = isinstance(pdf, str)
        try:
            pdf_reader = PdfReader(pdf)
            source_name = os.path.basename(pdf) if is_file_path else pdf.name

            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:  # Ensure there is text on the page
                    documents.append(Document(
                        page_content=text,
                        metadata={'source': source_name, 'page': page_num + 1}
                    ))
        except Exception as e:
            pdf_name = os.path.basename(pdf) if is_file_path else getattr(pdf, 'name', 'Unknown File')
            print(f"Error reading or processing PDF '{pdf_name}': {e}")
            continue
    return documents


def get_text_chunks(documents: List[Document]) -> List[Document]:
    """
    Splits a list of Documents into smaller, more manageable chunks.

    This is a key step for vector embedding, as it allows the model to find
    more specific and relevant pieces of information during a similarity search.
    The metadata from the parent document is automatically preserved in each chunk.

    Args:
        documents: A list of Document objects (typically from get_pdf_text_and_metadata).

    Returns:
        A list of smaller Document chunks, ready for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Increased for potentially denser academic text
        chunk_overlap=250   # Overlap helps maintain context between chunks
    )
    chunks = text_splitter.split_documents(documents)
    return chunks