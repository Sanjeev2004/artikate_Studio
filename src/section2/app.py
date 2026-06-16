import streamlit as st
import os
import sys

# Ensure the root directory is available in the path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.section2.pipeline import RAGPipeline

st.set_page_config(page_title="Legal RAG Chat", page_icon="⚖️", layout="wide")

# Initialize session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_ingested" not in st.session_state:
    st.session_state.documents_ingested = False

# Sidebar for Document Upload
with st.sidebar:
    st.title("📂 Upload Documents")
    st.write("Upload your legal PDFs to chat with them.")
    
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Save uploaded files to a temporary directory
                upload_dir = "data/uploads"
                os.makedirs(upload_dir, exist_ok=True)
                
                # Clear old uploads if necessary, but here we just overwrite/add
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(upload_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Ingest the directory
                st.session_state.pipeline.ingest_directory(upload_dir)
                st.session_state.documents_ingested = True
                st.success(f"Successfully processed {len(uploaded_files)} documents!")
        else:
            st.warning("Please upload at least one PDF.")

    st.divider()
    st.markdown("""
    ### About
    This is a Production-Grade RAG Pipeline built for the Artikate Studio Assessment.
    It features:
    - Local `all-MiniLM-L6-v2` embeddings
    - FAISS Vector Store
    - Paragraph-aware chunking
    - 3-tier hallucination mitigation
    """)

# Main Chat Area
st.title("⚖️ Legal RAG Chat Assistant")

if not st.session_state.documents_ingested:
    st.info("👈 Please upload and process some PDF documents in the sidebar to start chatting.")
else:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for idx, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {idx+1}: {source['document']} (Page {source['page']})**")
                        st.caption(f'"{source["chunk"]}"')

    # React to user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                result = st.session_state.pipeline.query(prompt)
                
                response_text = result["answer"]
                confidence = result["confidence"]
                sources = result["sources"]
                
                st.markdown(response_text)
                
                if sources:
                    st.caption(f"Confidence Score: {confidence:.2f}")
                    with st.expander("View Sources"):
                        for idx, source in enumerate(sources):
                            st.markdown(f"**Source {idx+1}: {source['document']} (Page {source['page']})**")
                            st.caption(f'"{source["chunk"]}"')
                            
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "sources": sources,
            "confidence": confidence
        })
