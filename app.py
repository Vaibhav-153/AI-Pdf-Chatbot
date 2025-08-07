# app.py - ManthanAI: Minimalist Version (No Custom CSS, Default Streamlit Styling)

import os
import streamlit as st
from dotenv import load_dotenv

# --- Backend Imports ---
from backend.ingestion import process_uploaded_files, get_text_chunks
from backend.embedding import create_hybrid_retriever
from backend.chat import get_rag_response

# --- CONFIGURATION ---
st.set_page_config(
    page_title="ManthanAI - Your AI Document Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INIT ---
def initialize_session_state():
    if "api_keys_loaded" not in st.session_state:
        load_dotenv()
        st.session_state.google_api_key = os.getenv("GEMINI_API_KEY")
        st.session_state.cohere_api_key = os.getenv("COHERE_API_KEY")
        st.session_state.api_keys_loaded = True

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "ai", "content": "Hello! I am ManthanAI. Upload documents to begin your analysis."}]
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "processed_doc_names" not in st.session_state:
        st.session_state.processed_doc_names = []
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "Hybrid"

# --- FILE HANDLING ---
def handle_file_upload():
    if st.session_state.get('file_uploader_key'):
        with st.spinner("Processing uploaded documents..."):
            try:
                uploaded_files = st.session_state.file_uploader_key
                raw_docs = process_uploaded_files(uploaded_files)
                text_chunks = get_text_chunks(raw_docs)
                if not text_chunks:
                    st.error("No text extracted. Please check your files.")
                    return
                st.session_state.retriever = create_hybrid_retriever(
                    text_chunks,
                    st.session_state.google_api_key,
                    st.session_state.cohere_api_key
                )
                st.session_state.processed_doc_names = [f.name for f in uploaded_files]
                st.success(f"Processed {len(st.session_state.processed_doc_names)} documents successfully.")
                st.session_state.chat_history.append({"role": "ai", "content": "Documents ready. Ask any question below."})
            except Exception as e:
                st.error(f"Processing error: {e}")
                st.session_state.retriever = None
                st.session_state.processed_doc_names = []

# --- SIDEBAR ---
def build_sidebar():
    with st.sidebar:
        st.title("ManthanAI")
        st.caption("AI-powered document insights and analysis tool.")

        st.header("Step 1: Choose Search Mode")
        st.radio(
            "AI Mode:", ("Hybrid", "PDF-Only"),
            key="search_mode", horizontal=True,
            help="Hybrid: Uses AI + docs | PDF-Only: Answers strictly from uploaded files"
        )

        st.header("Step 2: Upload Your Files")
        st.file_uploader(
            "Upload PDFs, DOCX, or PPTX", type=["pdf", "docx", "pptx"],
            accept_multiple_files=True, key='file_uploader_key', on_change=handle_file_upload
        )

        if st.session_state.processed_doc_names:
            st.subheader("Uploaded Files")
            for name in st.session_state.processed_doc_names:
                st.markdown(f"- {name}")

        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                if key not in ['api_keys_loaded', 'google_api_key', 'cohere_api_key']:
                    del st.session_state[key]
            initialize_session_state()
            st.rerun()

# --- CHAT HISTORY ---
def display_chat_history():
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "ai" else "🧑"):
            st.markdown(msg["content"])
            if msg["role"] == "ai" and msg.get("sources"):
                with st.expander("View Sources"):
                    for doc in msg["sources"]:
                        source_name = doc.metadata.get('source', 'Unknown')
                        page_num = doc.metadata.get('page') or doc.metadata.get('slide', 'N/A')
                        st.info(f"**Source:** {source_name} | Page/Slide: {page_num}")
                        st.markdown(f"> {doc.page_content[:300].strip()}...")

# --- MAIN ---
def main():
    initialize_session_state()

    if not st.session_state.google_api_key or not st.session_state.cohere_api_key:
        st.error("Missing API keys. Please set them in your .env file.")
        st.stop()

    build_sidebar()
    st.header("💬 Chat with Your Documents")
    display_chat_history()

    greetings = ['hi', 'hello', 'hey']
    if user_input := st.chat_input("Ask a question about your documents..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if user_input.lower().strip() in greetings:
            st.session_state.chat_history.append({"role": "ai", "content": "Hello! How can I help with your files today?"})
        elif not st.session_state.retriever:
            st.warning("Please upload documents first.")
        else:
            with st.spinner("Generating answer..."):
                response = get_rag_response(
                    user_input,
                    st.session_state.retriever,
                    st.session_state.google_api_key,
                    st.session_state.search_mode
                )
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": response["output_text"],
                    "sources": response.get("source_documents", [])
                })
        st.rerun()

if __name__ == "__main__":
    main()
