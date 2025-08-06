# pdf_chatbot/backend/pdf_utils.py

"""
Module for advanced PDF utility functions.

This script implements the capabilities outlined in Rule #4 of the persona,
providing tools for summarization, key point extraction, concept explanation,
keyword finding, and generating meeting minutes. These functions operate on
the full text of a selected document.
"""

from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

# A helper function to combine all text from a list of Document objects
def _get_full_text_from_docs(docs: List[Document]) -> str:
    """Concatenates the page_content of all documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])

def summarize_pdf(docs: List[Document], api_key: str) -> str:
    """
    Generates a comprehensive summary of the provided document text. (Rule #4.A)

    Args:
        docs: A list of Document objects representing the pages of a single PDF.
        api_key: The Google API key.

    Returns:
        A string containing the summary of the document.
    """
    full_text = _get_full_text_from_docs(docs)
    if not full_text:
        return "The document appears to be empty or could not be read."

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
    
    prompt = f"""
    Based on the following document text, please create a concise and comprehensive summary.
    Structure the summary logically, using section headers if appropriate.

    Document Text:
    ---
    {full_text[:20000]} 
    ---

    Summary:
    """
    
    summary = model.invoke(prompt).content
    return summary

def extract_key_points(docs: List[Document], api_key: str) -> str:
    """
    Extracts key points from a document as a bulleted list. (Rule #4.B)

    Args:
        docs: A list of Document objects for a single PDF.
        api_key: The Google API key.

    Returns:
        A markdown-formatted string of key points.
    """
    full_text = _get_full_text_from_docs(docs)
    if not full_text:
        return "The document appears to be empty or could not be read."

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.2)
    
    prompt = f"""
    Analyze the following document text and extract the most important key points and insights.
    Present them as a clear, bullet-point list in markdown format.

    Document Text:
    ---
    {full_text[:20000]}
    ---

    Key Points:
    """
    
    key_points = model.invoke(prompt).content
    return key_points

def explain_concept(docs: List[Document], concept: str, api_key: str) -> str:
    """
    Explains a specific concept using the document as context. (Rule #4.C)

    Args:
        docs: A list of Document objects for context.
        concept: The term or concept the user wants explained.
        api_key: The Google API key.

    Returns:
        A formatted explanation of the concept.
    """
    full_text = _get_full_text_from_docs(docs)
    if not full_text:
        return "The document appears to be empty or could not be read."

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
    
    prompt = f"""
    You are a helpful tutor. Based on the context from the document text provided,
    explain the following concept in a simple and clear way: "{concept}"

    Use the following structure for your explanation:
    - **Definition:** (Provide a clear definition)
    - **Example:** (Give a simple example, preferably from the document context if available)
    - **In Context:** (Explain how the concept is used or why it's important within the document)

    Document Text:
    ---
    {full_text[:20000]}
    ---

    Explanation for "{concept}":
    """
    
    explanation = model.invoke(prompt).content
    return explanation

def find_keywords(docs: List[Document], api_key: str) -> str:
    """
    Identifies and lists key terms from a document. (Rule #4.D)

    Args:
        docs: A list of Document objects for a single PDF.
        api_key: The Google API key.

    Returns:
        A markdown-formatted list of keywords.
    """
    full_text = _get_full_text_from_docs(docs)
    if not full_text:
        return "The document appears to be empty or could not be read."

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.1)
    
    prompt = f"""
    Analyze the following document text and identify the top 10-15 most important keywords, phrases, or names.
    Present them as a bulleted list.

    Document Text:
    ---
    {full_text[:20000]}
    ---

    Key Terms:
    """
    
    keywords = model.invoke(prompt).content
    return keywords

def generate_meeting_minutes(docs: List[Document], api_key: str) -> str:
    """
    Summarizes a document into a structured meeting minutes format. (Rule #4.E)

    Args:
        docs: A list of Document objects for a single PDF (presumably meeting notes).
        api_key: The Google API key.

    Returns:
        A structured summary formatted as meeting minutes.
    """
    full_text = _get_full_text_from_docs(docs)
    if not full_text:
        return "The document appears to be empty or could not be read."

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.1)
    
    prompt = f"""
    You are an expert at creating meeting summaries. Analyze the following document, which contains
    meeting notes or a transcript. Extract the key information and format it into a structured
    "Meeting Minutes" report using the following headers in markdown:
    
    - **Participants:**
    - **Agenda:**
    - **Decisions Made:**
    - **Action Items:** (Include who is responsible and deadlines if mentioned)

    If any of this information is not present, indicate "Not specified."

    Document Text:
    ---
    {full_text[:20000]}
    ---

    Meeting Minutes:
    """
    
    minutes = model.invoke(prompt).content
    return minutes