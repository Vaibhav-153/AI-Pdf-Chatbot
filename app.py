# pdf_chatbot/app.py

import os
import datetime
from collections import defaultdict
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai

# --- Backend Imports ---
# We only import what's necessary for the core Q&A feature.
from backend.ingestion import get_pdf_text_and_metadata, get_text_chunks
from backend.embedding import get_vector_store, VECTOR_STORE_PATH
from backend.chat import handle_user_query

# --- Helper Functions ---

def process_and_display_chat(question: str, api_key: str, answer_mode: str):
    """
    A centralized function to handle a user's question, get the AI response,
    and update the chat history and display.
    """
    # Add the user's question to the chat history and display it
    st.session_state.chat_history.append({"role": "user", "content": question, "sources": ""})
    with st.chat_message("user"):
        st.markdown(question)
    
    # Get and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = handle_user_query(question, api_key, answer_mode)
            answer = response.get("output_text", "An error occurred.")
            sources = consolidate_sources(response.get("source_documents", []))
            
            st.markdown(answer)
            if sources:
                st.caption(f"Sources: {sources}")
            
            # Add the assistant's response to the chat history
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})

def consolidate_sources(source_docs):
    """Consolidates source documents for clean display."""
    if not source_docs: return ""
    consolidated = defaultdict(list)
    for doc in source_docs:
        source_name = doc.metadata.get('source', 'Unknown File')
        page_num = doc.metadata.get('page', 'N/A')
        if page_num not in consolidated[source_name]: consolidated[source_name].append(page_num)
    output_parts = [f"**{source}** (Pages: {', '.join(map(str, sorted(pages)))})" for source, pages in consolidated.items()]
    return " | ".join(output_parts)

def generate_pdf_report(chat_history):
    """Generates a PDF report from chat history, fixing the unicode error."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Chat Session Report', 0, 1, 'C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 8, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
    pdf.ln(10)

    for message in chat_history:
        role = message["role"].capitalize()
        pdf.set_font("Arial", 'B', 12)
        pdf.multi_cell(0, 8, f"{role}:")
        
        pdf.set_font("Arial", '', 11)
        # FIX for FPDFException: Encode text to a format FPDF can handle
        content = message["content"].encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, content)
        
        if role == "Assistant" and message["sources"]:
            pdf.set_font("Arial", 'I', 9)
            sources = message["sources"].replace("**", "")
            sources_cleaned = sources.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, f"Sources: {sources_cleaned}")
        
        pdf.ln(5)

    return pdf.output(dest='S').encode('latin-1')


# --- Main Application UI ---
def main():
    load_dotenv()
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("API Key not found. Please set `GEMINI_API_KEY` in your .env file.")
            st.stop()
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Error configuring API: {e}")
        st.stop()
        
    st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="centered")

    # --- Session State Initialization ---
    if "chat_history" not in st.session_state: 
        st.session_state.chat_history = []
    
    # --- Sidebar ---
    with st.sidebar:
        st.title("📄 AI Assistant")
        st.markdown("---")
        
        st.header("1. Upload Documents")
        uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf", label_visibility="collapsed")
        
        if st.button("Process Documents", use_container_width=True, type="primary"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    raw_docs = get_pdf_text_and_metadata(uploaded_files)
                    chunks = get_text_chunks(raw_docs)
                    get_vector_store(chunks, api_key)
                st.success("✅ Documents Processed!")
            else:
                st.warning("⚠️ Please upload at least one PDF.")
        
        st.markdown("---")
        st.header("2. Chat Settings")
        st.radio("Answer Mode:", ["PDF-Only", "Hybrid"], horizontal=True, key="answer_mode")
        
        if st.session_state.chat_history:
            st.markdown("---")
            st.header("3. Export")
            pdf_report = generate_pdf_report(st.session_state.chat_history)
            st.download_button("Download Chat PDF", data=pdf_report, file_name="chat_report.pdf", mime="application/pdf", use_container_width=True)

    # --- Main Chat Area ---
    st.header("Conversational Q&A")
    is_ready = os.path.exists(VECTOR_STORE_PATH)
    
    # Display chat history. This will now scroll correctly.
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message["sources"]:
                st.caption(f"Sources: {message['sources']}")

    # --- Combined Input Area at the bottom ---
    st.divider()
    col_mic, col_chat = st.columns([1, 20])

    with col_mic:
        # The voice recorder sits neatly next to the chat input
        audio_info = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key='recorder', just_once=True)
    
    with col_chat:
        # The text input
        user_question = st.chat_input("Ask a question...", disabled=not is_ready)

    # --- Logic to handle inputs ---
    
    # Process voice input
    if audio_info and audio_info['bytes']:
        with st.spinner("Transcribing and processing..."):
            try:
                with open("temp_audio.wav", "wb") as f: f.write(audio_info['bytes'])
                audio_file = genai.upload_file(path="temp_audio.wav")
                model = genai.GenerativeModel('models/gemini-1.5-flash')
                response = model.generate_content(["Transcribe this audio.", audio_file])
                os.remove("temp_audio.wav")
                transcribed_text = response.text.strip()
                
                process_and_display_chat(transcribed_text, api_key, st.session_state.answer_mode)
                st.rerun()
            except Exception as e:
                st.error(f"Error during transcription: {e}")

    # Process text input
    if user_question:
        process_and_display_chat(user_question, api_key, st.session_state.answer_mode)

if __name__ == "__main__":
    main()