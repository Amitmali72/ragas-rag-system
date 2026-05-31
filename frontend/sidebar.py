"""
Sidebar component for the Knowledge Copilot application.

Handles document upload, ingestion pipeline execution, retrieval settings,
model configuration, and session state management. Renders a premium-styled
sidebar with all controls the user needs to configure the RAG pipeline.
"""

import streamlit as st
from typing import Dict, List, Any

# --- Backend imports ---
from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_SEMANTIC,
    TOP_K_BM25,
    GROQ_MODELS,
    DEFAULT_GROQ_MODEL,
    GROQ_API_KEY,
    SUPPORTED_FILE_TYPES,
)
from ingestion.file_parser import parse_file
from ingestion.text_cleaner import clean_documents
from ingestion.chunker import create_chunks
from embeddings.embedding_manager import (
    get_embedding_model,
    create_faiss_index,
    save_faiss_index,
    load_faiss_index,
)
from retrieval.keyword_retriever import KeywordRetriever


# ---------------------------------------------------------------------------
# Custom CSS for the sidebar
# ---------------------------------------------------------------------------
SIDEBAR_CSS = """
<style>
    /* ── Sidebar wrapper ─────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0f1a 0%, #141829 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.15);
    }

    /* ── Section headers inside sidebar ──────────────────────── */
    [data-testid="stSidebar"] .sidebar-section-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #9d99ff;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.2rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid rgba(108, 99, 255, 0.2);
    }

    /* ── File list chips ─────────────────────────────────────── */
    .file-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(108, 99, 255, 0.1);
        border: 1px solid rgba(108, 99, 255, 0.25);
        border-radius: 8px;
        padding: 0.3rem 0.7rem;
        margin: 0.2rem 0;
        font-size: 0.8rem;
        color: #c9c5ff;
        width: 100%;
        box-sizing: border-box;
    }
    .file-chip .file-size {
        margin-left: auto;
        color: #7a78a0;
        font-size: 0.72rem;
    }

    /* ── Process button ──────────────────────────────────────── */
    [data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #6C63FF 0%, #4e47c9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.25s ease;
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover {
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.45);
        transform: translateY(-1px);
    }

    /* ── Logo / branding area ────────────────────────────────── */
    .sidebar-brand {
        text-align: center;
        padding: 1rem 0 0.8rem 0;
        border-bottom: 1px solid rgba(108, 99, 255, 0.15);
        margin-bottom: 0.6rem;
    }
    .sidebar-brand h1 {
        font-size: 1.4rem;
        margin: 0;
        background: linear-gradient(135deg, #6C63FF, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-brand p {
        font-size: 0.75rem;
        color: #7a78a0;
        margin: 0.2rem 0 0 0;
    }

    /* ── Stats row ───────────────────────────────────────────── */
    .sidebar-stat-row {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
    }
    .sidebar-stat-row .stat-label {
        color: #7a78a0;
        font-size: 0.78rem;
    }
    .sidebar-stat-row .stat-value {
        color: #c9c5ff;
        font-weight: 600;
        font-size: 0.78rem;
    }
</style>
"""


def _format_file_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string (KB / MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _init_session_state() -> None:
    """Ensure every session-state key the sidebar relies on exists."""
    defaults: Dict[str, Any] = {
        "documents_processed": False,
        "chunks": [],
        "faiss_index": None,
        "keyword_retriever": None,
        "file_names": [],
        "num_chunks": 0,
        "num_documents": 0,
        "messages": [],
        "uploaded_files_info": [],
        "raw_documents": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _run_ingestion_pipeline(
    uploaded_files: List[Any],
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """
    Execute the full ingestion pipeline:
      parse ➜ clean ➜ chunk ➜ embed ➜ build keyword index.

    Updates session_state with the processed artefacts.
    """
    progress = st.sidebar.progress(0, text="Starting ingestion…")

    # ── Step 1: Parse uploaded files ────────────────────────────
    progress.progress(10, text="📄 Parsing documents…")
    all_documents: List[Dict] = []
    file_names: List[str] = []

    for uploaded_file in uploaded_files:
        try:
            docs = parse_file(uploaded_file)
            all_documents.extend(docs)
            file_names.append(uploaded_file.name)
        except Exception as exc:
            st.sidebar.error(f"Failed to parse **{uploaded_file.name}**: {exc}")

    if not all_documents:
        progress.empty()
        st.sidebar.error("No documents could be parsed. Please check your files.")
        return

    # ── Step 2: Clean documents ─────────────────────────────────
    progress.progress(30, text="🧹 Cleaning text…")
    cleaned_documents = clean_documents(all_documents)

    # ── Step 3: Create chunks ───────────────────────────────────
    progress.progress(50, text="✂️ Chunking documents…")
    chunks = create_chunks(cleaned_documents, chunk_size, chunk_overlap)

    if not chunks:
        progress.empty()
        st.sidebar.error("Chunking produced zero chunks. Try different settings.")
        return

    # ── Step 4: Create FAISS index ──────────────────────────────
    progress.progress(70, text="🔍 Building vector index…")
    faiss_index = create_faiss_index(chunks)

    # ── Step 5: Build keyword retriever ─────────────────────────
    progress.progress(85, text="📚 Building keyword index…")
    keyword_retriever = KeywordRetriever(chunks)

    # ── Step 6: Save artefacts to session_state ─────────────────
    progress.progress(95, text="💾 Saving artefacts…")
    st.session_state["documents_processed"] = True
    st.session_state["chunks"] = chunks
    st.session_state["faiss_index"] = faiss_index
    st.session_state["keyword_retriever"] = keyword_retriever
    st.session_state["file_names"] = file_names
    st.session_state["num_chunks"] = len(chunks)
    st.session_state["num_documents"] = len(set(file_names))
    st.session_state["raw_documents"] = cleaned_documents

    progress.progress(100, text="✅ Ingestion complete!")
    st.sidebar.success(
        f"Processed **{len(file_names)}** file(s) → **{len(chunks)}** chunks"
    )


def _clear_all() -> None:
    """Reset every session-state key to its default value."""
    keys_to_clear = [
        "documents_processed",
        "chunks",
        "faiss_index",
        "keyword_retriever",
        "file_names",
        "num_chunks",
        "num_documents",
        "messages",
        "uploaded_files_info",
        "raw_documents",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_sidebar() -> Dict[str, Any]:
    """
    Render the full sidebar and return a settings dictionary consumed by
    the rest of the application.

    Returns:
        Dict with keys: chunk_size, chunk_overlap, top_k, model_name,
        api_key, uploaded_files.
    """
    _init_session_state()

    with st.sidebar:
        # ── Inject CSS ──────────────────────────────────────────
        st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

        # ── Brand / logo ────────────────────────────────────────
        st.markdown(
            """
            <div class="sidebar-brand">
                <h1>🧠 Knowledge Copilot</h1>
                <p>Hybrid RAG • Semantic + Keyword Search</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Section 1: Document Upload ──────────────────────────
        st.markdown(
            '<p class="sidebar-section-header">📄 Document Upload</p>',
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Drop your documents here",
            type=SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Supports PDF, DOCX, and TXT files up to 200 MB total.",
        )

        # ── Section 2: Uploaded Documents ───────────────────────
        if uploaded_files:
            st.markdown(
                '<p class="sidebar-section-header">📁 Uploaded Documents</p>',
                unsafe_allow_html=True,
            )
            for uf in uploaded_files:
                size_str = _format_file_size(uf.size)
                st.markdown(
                    f'<div class="file-chip">'
                    f"📎 {uf.name}"
                    f'<span class="file-size">{size_str}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── Section 3: Retrieval Settings ───────────────────────
        st.markdown(
            '<p class="sidebar-section-header">⚙️ Retrieval Settings</p>',
            unsafe_allow_html=True,
        )
        chunk_size: int = st.slider(
            "Chunk Size (tokens)",
            min_value=200,
            max_value=1000,
            value=CHUNK_SIZE,
            step=50,
            help="Number of tokens per text chunk during ingestion.",
        )
        chunk_overlap: int = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=200,
            value=CHUNK_OVERLAP,
            step=25,
            help="Overlap between consecutive chunks to preserve context.",
        )
        top_k: int = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=20,
            value=TOP_K_SEMANTIC,
            step=1,
            help="Number of chunks to retrieve for each query.",
        )

        # ── Section 4: Model Settings ──────────────────────────
        st.markdown(
            '<p class="sidebar-section-header">🤖 Model Settings</p>',
            unsafe_allow_html=True,
        )

        # Build display list for the selectbox and resolve default index
        model_options: List[str] = list(GROQ_MODELS.keys())
        default_display = "Llama 3.3 70B"
        for disp_name, api_id in GROQ_MODELS.items():
            if api_id == DEFAULT_GROQ_MODEL:
                default_display = disp_name
                break

        default_index: int = (
            model_options.index(default_display)
            if default_display in model_options
            else 0
        )
        model_name: str = st.selectbox(
            "Groq Model",
            options=model_options,
            index=default_index,
            help="Select the LLM served via Groq for answer generation.",
        )

        api_key: str = st.text_input(
            "Groq API Key",
            value=GROQ_API_KEY or "",
            type="password",
            help="Enter your Groq API key. Stored only in this session.",
        )

        # ── Action Buttons ──────────────────────────────────────
        st.markdown("---")
        col_process, col_clear = st.columns(2)

        with col_process:
            process_clicked = st.button(
                "🔄 Process",
                use_container_width=True,
                type="primary",
                disabled=not uploaded_files,
            )

        with col_clear:
            clear_clicked = st.button(
                "🗑️ Clear All",
                use_container_width=True,
            )

        # ── Pipeline execution ──────────────────────────────────
        if process_clicked and uploaded_files:
            _run_ingestion_pipeline(uploaded_files, chunk_size, chunk_overlap)

        if clear_clicked:
            _clear_all()
            st.rerun()

        # ── Processing status panel ─────────────────────────────
        if st.session_state.get("documents_processed"):
            st.markdown(
                '<p class="sidebar-section-header">📊 Pipeline Status</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="sidebar-stat-row">
                    <span class="stat-label">Documents</span>
                    <span class="stat-value">{st.session_state['num_documents']}</span>
                </div>
                <div class="sidebar-stat-row">
                    <span class="stat-label">Chunks</span>
                    <span class="stat-value">{st.session_state['num_chunks']}</span>
                </div>
                <div class="sidebar-stat-row">
                    <span class="stat-label">Index</span>
                    <span class="stat-value">FAISS ✅</span>
                </div>
                <div class="sidebar-stat-row">
                    <span class="stat-label">BM25</span>
                    <span class="stat-value">Ready ✅</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Return settings dict ────────────────────────────────────
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "model_name": GROQ_MODELS.get(model_name, model_name),
        "api_key": api_key,
        "uploaded_files": uploaded_files or [],
    }
