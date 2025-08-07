# backend/chat.py - Final Version with Improved Hybrid Prompt

from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Prompt Templates ---

PDF_ONLY_PROMPT = """
You are ManthanAI, a precise document assistant.
Your task is to answer the user's question based *only* on the provided document context.
If the information is not present in the context, you must state: "I couldn't find this information in the uploaded documents."
Do not use any external knowledge. Do not make up answers.

Context:
{context}

Question:
{question}

Answer:
"""

# NEW: A much more robust prompt for Hybrid mode
HYBRID_PROMPT = """
You are ManthanAI, an expert assistant. Your goal is to be as helpful as possible.
A user has asked a question. You have been provided with some context from their documents.

Your instructions are:
1. First, critically evaluate the provided context. Is it relevant and does it actually help answer the user's question?
2. If the context is relevant, answer the question using ONLY the information from the context. At the end of your answer, cite the source (e.g., [Source: report.pdf, page 2]).
3. If the context is NOT relevant or does not contain the answer, you MUST IGNORE IT COMPLETELY. In this case, answer the question using your own general knowledge. Do not mention the documents or context at all.

Context:
{context}

Question:
{question}

Answer:
"""

def get_rag_chain(retriever, llm: ChatGoogleGenerativeAI, mode: str) -> any:
    """
    Builds a RAG (Retrieval-Augmented Generation) chain using LCEL,
    based on the selected search mode.
    """
    prompt_template = HYBRID_PROMPT if mode == "Hybrid" else PDF_ONLY_PROMPT
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    def format_docs(docs: List[Document]) -> str:
        """Concatenates document content for the prompt context."""
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'N/A')}, "
            f"Page/Slide: {doc.metadata.get('page') or doc.metadata.get('slide', 'N/A')}\n"
            f"Content: {doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def get_rag_response(user_question: str, retriever, gemini_api_key: str, mode: str) -> Dict[str, Any]:
    """
    Gets a response from the RAG chain.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.2,
        convert_system_message_to_human=True
    )
    source_documents = retriever.invoke(user_question)
    rag_chain = get_rag_chain(retriever, llm, mode)
    answer = rag_chain.invoke(user_question)
    return {"output_text": answer, "source_documents": source_documents}