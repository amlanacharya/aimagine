import streamlit as st
from pathlib import Path
import sys

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.services.data_ingestion import DataIngestionService
from src.services.embedding_service import EmbeddingService
from src.services.faiss_service import FAISSService

def initialize_services():
    """Initialize all required services"""
    return {
        'embedding': EmbeddingService(),
        'faiss': FAISSService(),
        'ingestion': DataIngestionService()
    }

def setup_page():
    """Configure the Streamlit page"""
    st.set_page_config(
        page_title="Airline Knowledge Base Chat",
        page_icon="✈️",
        layout="wide"
    )
    st.title("✈️ Airline Assistant")

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'services' not in st.session_state:
        st.session_state.services = initialize_services()

def main():
    setup_page()
    initialize_session_state()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("How can I help you?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            # Generate query embedding
            query_embedding = st.session_state.services['embedding'].generate_query_embedding(prompt)
            
            # Search for relevant information
            results = st.session_state.services['faiss'].search(query_embedding, k=3)
            
            # Format response
            response = "Here's what I found:\n\n"
            for i, result in enumerate(results, 1):
                response += f"{i}. {result['text']}\n\n"
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 