"""
Knowledge Copilot – Main Streamlit Application
================================================

Entry point for the Hybrid RAG application.  Composes the sidebar,
chat interface, and evaluation dashboard into a cohesive, premium-styled
single-page application.

Run with:
    streamlit run app.py
"""

import streamlit as st
from typing import Dict, Any

# --- Frontend component imports ---
from frontend.sidebar import render_sidebar
from frontend.chat_interface import render_chat_interface
from frontend.metrics_display import render_metrics, render_document_stats
from frontend.evaluation_dashboard import render_evaluation_dashboard


# ──────────────────────────────────────────────────────────────────────────────
# Page configuration (MUST be the first Streamlit command)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "**Knowledge Copilot** – A Hybrid RAG system combining semantic "
            "and keyword retrieval for accurate, cited answers."
        ),
    },
)


# ──────────────────────────────────────────────────────────────────────────────
# Global custom CSS – premium dark theme
# ──────────────────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
    /* ── Google Font import ──────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Base ────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide default Streamlit header / footer ──────────────── */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }

    /* ── Main container ──────────────────────────────────────── */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* ── Gradient top bar ────────────────────────────────────── */
    .app-header {
        background: linear-gradient(135deg, #6C63FF 0%, #4e47c9 50%, #2d2a6e 100%);
        border-radius: 14px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 30px rgba(108, 99, 255, 0.25);
    }
    .app-header .header-left h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
    }
    .app-header .header-left p {
        margin: 0.2rem 0 0 0;
        font-size: 0.82rem;
        color: rgba(255, 255, 255, 0.7);
    }
    .app-header .header-right {
        display: flex;
        gap: 0.8rem;
    }
    .header-badge {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.72rem;
        color: rgba(255, 255, 255, 0.85);
    }

    /* ── Card containers ─────────────────────────────────────── */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    /* ── Custom scrollbar ────────────────────────────────────── */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(108, 99, 255, 0.35);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(108, 99, 255, 0.55);
    }

    /* ── Button global styling ───────────────────────────────── */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.25s ease;
        border: 1px solid rgba(108, 99, 255, 0.2);
    }
    .stButton > button:hover {
        border-color: rgba(108, 99, 255, 0.5);
        box-shadow: 0 2px 12px rgba(108, 99, 255, 0.15);
    }

    /* ── Expander styling ────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #c9c5ff;
        border-radius: 8px;
    }

    /* ── Input widgets ───────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        border-radius: 8px;
    }

    /* ── Slider ──────────────────────────────────────────────── */
    .stSlider > div > div > div > div {
        color: #6C63FF;
    }

    /* ── Footer ──────────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 1.2rem 0 0.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid rgba(108, 99, 255, 0.1);
        font-size: 0.75rem;
        color: #5a577a;
    }
    .app-footer a {
        color: #6C63FF;
        text-decoration: none;
    }
    .app-footer a:hover {
        text-decoration: underline;
    }

    /* ── Warning / info box tweaks ───────────────────────────── */
    .stAlert {
        border-radius: 10px;
    }

    /* ── Chat input ──────────────────────────────────────────── */
    [data-testid="stChatInput"] textarea {
        border-radius: 10px;
    }

    /* ── Smooth transitions ──────────────────────────────────── */
    *, *::before, *::after {
        transition-property: background-color, border-color, box-shadow;
        transition-duration: 0.2s;
        transition-timing-function: ease;
    }
</style>
"""


def _init_app_state() -> None:
    """Initialise global session-state keys used across the app."""
    defaults: Dict[str, Any] = {
        "messages": [],
        "documents_processed": False,
        "chunks": [],
        "faiss_index": None,
        "keyword_retriever": None,
        "file_names": [],
        "num_chunks": 0,
        "num_documents": 0,
        "last_metrics": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_header() -> None:
    """Render the gradient header bar."""
    docs_status = (
        "✅ Ready" if st.session_state.get("documents_processed") else "⏳ No docs"
    )
    num_chunks = st.session_state.get("num_chunks", 0)

    st.markdown(
        f"""
        <div class="app-header">
            <div class="header-left">
                <h1>🧠 Knowledge Copilot</h1>
                <p>Hybrid RAG · Semantic + Keyword Retrieval · Powered by Groq</p>
            </div>
            <div class="header-right">
                <span class="header-badge">{docs_status}</span>
                <span class="header-badge">📦 {num_chunks} chunks</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    """Render the page footer."""
    st.markdown(
        """
        <div class="app-footer">
            <strong>Knowledge Copilot</strong> · Hybrid RAG System ·
            Built with
            <a href="https://streamlit.io" target="_blank">Streamlit</a>,
            <a href="https://groq.com" target="_blank">Groq</a>,
            FAISS & BM25
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Application entry point."""
    # Inject global CSS
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Initialise state
    _init_app_state()

    # ── Sidebar (returns settings dict) ─────────────────────────
    settings: Dict[str, Any] = render_sidebar()

    # ── API key check ───────────────────────────────────────────
    api_key = settings.get("api_key", "")
    if not api_key:
        st.warning(
            "🔑 **Groq API key not set.** Enter your key in the sidebar under "
            "*Model Settings* to enable answer generation.",
            icon="🔑",
        )

    # ── Header ──────────────────────────────────────────────────
    _render_header()

    # ── Document stats bar (compact) ────────────────────────────
    if st.session_state.get("documents_processed"):
        render_document_stats()

    # ── Main tabs ───────────────────────────────────────────────
    tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])

    with tab_chat:
        render_chat_interface(settings)

        # Show last-query metrics if available
        last_msgs = st.session_state.get("messages", [])
        if last_msgs:
            last_assistant = [m for m in last_msgs if m["role"] == "assistant"]
            if last_assistant and last_assistant[-1].get("metrics"):
                st.markdown("---")
                render_metrics(last_assistant[-1]["metrics"])

    with tab_eval:
        render_evaluation_dashboard(settings)

    # ── Footer ──────────────────────────────────────────────────
    _render_footer()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
