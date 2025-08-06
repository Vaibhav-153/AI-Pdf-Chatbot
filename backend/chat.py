# pdf_chatbot/backend/chat.py

"""
Module for handling the conversational Q&A logic.

This script implements the core of the chatbot's intelligence, particularly
Rule #1 (Answer Source Priority). It orchestrates the retrieval of relevant
document chunks and the generation of a context-aware answer. It also includes
the logic to fallback to general LLM knowledge if specified.
"""

from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document

from .embedding import load_vector_store # Relative import from the same package

# --- Prompt Templates ---

# This prompt strictly confines the model to the provided context.
PDF_ONLY_PROMPT_TEMPLATE = """
You are an intelligent, enterprise-grade PDF assistant.
Your task is to provide a detailed and accurate answer based *only* on the provided context from the user's documents.
If the answer is not found within the given context, you must respond with the exact phrase: "I'm sorry, I couldn’t find this information in the uploaded documents."
Do not use any external knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

# This prompt is for general knowledge, used as a fallback in Hybrid mode.
GENERAL_KNOWLEDGE_PROMPT_TEMPLATE = """
You are an intelligent, enterprise-grade assistant.
Answer the following question using your general knowledge.
Be comprehensive and clear in your explanation.
You are an intelligent assistant. Answer the question based on the provided context.
Crucially, you must respond in the same language as the user's question.

Context:
{context}

Question:
{question}

Answer:

Question:
{question}

Answer:
"""

def get_conversational_chain(api_key: str, prompt_template: str) -> any:
    """
    Initializes and returns a LangChain QA chain with a specific prompt.

    Args:
        api_key: The Google API key.
        prompt_template: The string template to guide the model's response.

    Returns:
        A configured LangChain question-answering chain.
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2, # Low temperature for factual, deterministic answers
        convert_system_message_to_human=True
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def handle_user_query(user_question: str, api_key: str, answer_mode: str) -> Dict[str, Any]:
    """
    Processes a user's question according to the specified answer mode.

    This function embodies Rule #1. It first attempts to answer from the PDFs.
    If it fails and the mode is 'Hybrid', it attempts a fallback to general knowledge.

    Args:
        user_question: The question asked by the user.
        api_key: The Google API key.
        answer_mode: The selected mode ("PDF-Only" or "Hybrid").

    Returns:
        A dictionary containing the answer text and a list of source documents.
    """
    # 1. Load the knowledge base
    vector_store = load_vector_store(api_key=api_key)
    if not vector_store:
        return {
            "output_text": "The knowledge base is not initialized. Please upload and process your documents.",
            "source_documents": []
        }

    # 2. Retrieve relevant documents from the vector store
    retrieved_docs = vector_store.similarity_search(user_question, k=5)

    # 3. Attempt to answer using only the PDF context
    pdf_chain = get_conversational_chain(api_key, PDF_ONLY_PROMPT_TEMPLATE)
    response = pdf_chain(
        {"input_documents": retrieved_docs, "question": user_question},
        return_only_outputs=False
    )
    pdf_answer = response.get("output_text", "")
    source_documents = response.get("input_documents", [])

    # 4. Check if the answer was found in the context
    # We check for the specific failure message we put in the prompt.
    if "couldn’t find this information" in pdf_answer:
        # If not found, check the user's selected mode
        if answer_mode == "Hybrid":
            print("Answer not found in documents. Falling back to general knowledge.")
            # Use a simpler model invocation for the general knowledge fallback
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.7)
            general_answer = model.invoke(user_question).content
            final_answer = (
                f"{general_answer}\n\n---\n"
                "*Disclaimer: This answer was generated using general knowledge as the information was not found in the uploaded documents.*"
            )
            return {"output_text": final_answer, "source_documents": []}
        else: # PDF-Only mode
            return {"output_text": pdf_answer, "source_documents": []}
    
    # 5. If the answer was found, return it with sources
    return {"output_text": pdf_answer, "source_documents": source_documents}