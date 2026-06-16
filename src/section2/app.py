import streamlit as st
import os
import sys
import time
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.env')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.section2.pipeline import RAGPipeline

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexAI — Legal Document Assistant",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Reset */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark Background */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%);
    color: #e2e8f0;
}

/* Hide default streamlit header */
#MainMenu, header, footer { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(13, 17, 23, 0.95);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
    backdrop-filter: blur(20px);
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8;
}

/* Sidebar Title */
.sidebar-brand {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2);
    margin-bottom: 1.5rem;
}

.sidebar-logo {
    font-size: 2.2rem;
    margin-bottom: 0.3rem;
}

.sidebar-title {
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

.sidebar-sub {
    font-size: 0.72rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.2rem;
}

/* Status Badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    width: 100%;
    margin: 0.5rem 0;
}

.status-active {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
}

.status-inactive {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fbbf24;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 6px currentColor;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(99, 102, 241, 0.05);
    border: 1.5px dashed rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 0.5rem;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 102, 241, 0.6);
    background: rgba(99, 102, 241, 0.08);
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.3px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4338ca, #6d28d9);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    transform: translateY(-1px);
}

/* Hero Header */
.hero-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1.2;
}

.hero-sub {
    font-size: 0.95rem;
    color: #64748b;
    margin-top: 0.5rem;
}

/* Welcome Screen */
.welcome-card {
    background: rgba(99, 102, 241, 0.05);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    max-width: 550px;
    margin: 3rem auto;
}

.welcome-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
}

.welcome-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.5rem;
}

.welcome-text {
    color: #64748b;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Feature Pills */
.feature-pills {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 1.5rem;
}

.pill {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.75rem;
    color: #818cf8;
    font-weight: 500;
}

/* Chat Messages */
.stChatMessage {
    background: transparent !important;
    border: none !important;
}

[data-testid="stChatMessageContent"] {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px);
    padding: 1rem 1.2rem !important;
    color: #e2e8f0 !important;
}

/* User message */
[data-testid="stChatMessageContent"]:has(+ [data-testid="stChatMessageAvatarUser"]) {
    background: rgba(99, 102, 241, 0.12) !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
}

/* Source cards */
.source-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
}

.source-title {
    color: #818cf8;
    font-weight: 600;
    font-size: 0.8rem;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.source-text {
    color: #64748b;
    font-size: 0.78rem;
    line-height: 1.5;
    font-style: italic;
}

/* Confidence bar */
.confidence-bar-bg {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 10px;
    height: 6px;
    margin-top: 0.6rem;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    transition: width 0.5s ease;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99, 102, 241, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* Section labels */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #475569;
    margin: 1.2rem 0 0.5rem 0;
}

/* Divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
    margin: 1rem 0;
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
    border-radius: 12px !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #818cf8 !important;
}

/* Success/Warning overrides */
.stAlert {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state or not hasattr(st.session_state.pipeline, "clear"):
    st.session_state.pipeline = RAGPipeline()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_ingested" not in st.session_state:
    st.session_state.documents_ingested = False
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">⚖</div>
        <div class="sidebar-title">LexAI</div>
        <div class="sidebar-sub">Legal Document Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    # LLM Status
    st.markdown('<div class="section-label">LLM Engine</div>', unsafe_allow_html=True)
    if os.environ.get("GROQ_API_KEY"):
        st.markdown('<div class="status-badge status-active"><div class="status-dot"></div> Groq · Llama 3.3 70B</div>', unsafe_allow_html=True)
    elif os.environ.get("OPENAI_API_KEY"):
        st.markdown('<div class="status-badge status-active"><div class="status-dot"></div> OpenAI · GPT-4o Mini</div>', unsafe_allow_html=True)
    elif os.environ.get("GEMINI_API_KEY"):
        st.markdown('<div class="status-badge status-active"><div class="status-dot"></div> Gemini 2.5 Flash</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-inactive"><div class="status-dot"></div> Local Fallback Mode</div>', unsafe_allow_html=True)
        st.caption("Add `GROQ_API_KEY` to your `.env` file for real LLM answers.")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Document Upload
    st.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if st.button("Process & Index Documents"):
        if uploaded_files:
            with st.spinner("Indexing documents..."):
                upload_dir = "data/uploads"
                # Clear previous uploaded PDF files from the local directory to avoid mixing old documents
                if os.path.exists(upload_dir):
                    for file in os.listdir(upload_dir):
                        if file.endswith(".pdf"):
                            file_path = os.path.join(upload_dir, file)
                            try:
                                os.unlink(file_path)
                            except Exception as e:
                                pass
                else:
                    os.makedirs(upload_dir, exist_ok=True)
                
                # Reset the vector store index/metadata and clear chat history
                if hasattr(st.session_state.pipeline, "clear"):
                    st.session_state.pipeline.clear()
                else:
                    st.session_state.pipeline = RAGPipeline()
                st.session_state.messages = []
                
                names = []
                for f in uploaded_files:
                    path = os.path.join(upload_dir, f.name)
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                    names.append(f.name)
                st.session_state.pipeline.ingest_directory(upload_dir)
                st.session_state.documents_ingested = True
                st.session_state.ingested_files = names
                st.success(f"Indexed {len(names)} document(s)!")
        else:
            st.warning("Please upload at least one PDF.")

    # Show indexed docs
    if st.session_state.ingested_files:
        st.markdown('<div class="section-label">Indexed</div>', unsafe_allow_html=True)
        for fname in st.session_state.ingested_files:
            st.markdown(f"<div style='font-size:0.78rem;color:#64748b;padding:2px 0'>📄 {fname}</div>", unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # About
    st.markdown('<div class="section-label">Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem;color:#475569;line-height:1.8'>
        Embeddings · all-MiniLM-L6-v2<br>
        Vector Store · FAISS<br>
        Chunking · Paragraph-aware<br>
        Hallucination · 3-tier guard<br>
        LLM · Llama 3.3 / GPT-4o / Gemini
    </div>
    """, unsafe_allow_html=True)

# ── Main Area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Legal Document Intelligence</div>
    <div class="hero-sub">Ask precise questions. Get cited answers. Zero hallucinations.</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.documents_ingested:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">📂</div>
        <div class="welcome-title">Upload your documents to begin</div>
        <div class="welcome-text">
            Upload your legal PDFs in the sidebar and click<br>
            <strong style='color:#818cf8'>Process & Index Documents</strong> to get started.
        </div>
        <div class="feature-pills">
            <span class="pill">PDF Contracts</span>
            <span class="pill">NDA Analysis</span>
            <span class="pill">Policy Review</span>
            <span class="pill">Clause Extraction</span>
            <span class="pill">Source Citations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("fallback"):
                st.warning("⚠️ LLM Generation failed (e.g. rate limit or API issue). Showing local fallback sentence extraction.")
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                conf = message.get("confidence", 0)
                bar_color = "#10b981" if conf > 0.6 else "#f59e0b" if conf > 0.35 else "#ef4444"
                st.markdown(
                    f"<div style='font-size:0.72rem;color:#475569;margin-top:0.5rem'>"
                    f"Confidence · {conf:.0%}"
                    f"<div class='confidence-bar-bg'><div class='confidence-bar-fill' style='width:{conf*100:.0f}%;background:linear-gradient(90deg,{bar_color},{bar_color}cc)'></div></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                with st.expander(f"Sources ({len(message['sources'])} retrieved)"):
                    for idx, src in enumerate(message["sources"]):
                        st.markdown(
                            f"<div class='source-card'>"
                            f"<div class='source-title'>📄 {src['document']} &nbsp;·&nbsp; Page {src['page']}</div>"
                            f"<div class='source-text'>\"{src['chunk'][:220]}...\"</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

    # Chat input
    if prompt := st.chat_input("Ask anything about your documents..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Searching and reasoning..."):
                time.sleep(0.3)  # slight delay for UX
                result = st.session_state.pipeline.query(prompt)

            answer = result["answer"]
            confidence = result["confidence"]
            sources = result["sources"]
            is_fallback = result.get("fallback", False)

            if is_fallback:
                st.warning("⚠️ LLM Generation failed (e.g. rate limit or API issue). Showing local fallback sentence extraction.")

            st.markdown(answer)

            if sources:
                bar_color = "#10b981" if confidence > 0.6 else "#f59e0b" if confidence > 0.35 else "#ef4444"
                st.markdown(
                    f"<div style='font-size:0.72rem;color:#475569;margin-top:0.5rem'>"
                    f"Confidence · {confidence:.0%}"
                    f"<div class='confidence-bar-bg'><div class='confidence-bar-fill' style='width:{confidence*100:.0f}%;background:linear-gradient(90deg,{bar_color},{bar_color}cc)'></div></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                with st.expander(f"Sources ({len(sources)} retrieved)"):
                    for idx, src in enumerate(sources):
                        st.markdown(
                            f"<div class='source-card'>"
                            f"<div class='source-title'>📄 {src['document']} &nbsp;·&nbsp; Page {src['page']}</div>"
                            f"<div class='source-text'>\"{src['chunk'][:220]}...\"</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "confidence": confidence,
            "fallback": is_fallback
        })
