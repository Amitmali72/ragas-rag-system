"""
Metrics display component for the Knowledge Copilot application.

Provides reusable functions to render latency metrics, retrieval statistics,
and overall document-level stats using Streamlit metric cards with
custom CSS styling.
"""

import streamlit as st
from typing import Dict, Any

from utils.helpers import format_time


# ---------------------------------------------------------------------------
# Custom CSS for the metrics cards
# ---------------------------------------------------------------------------
METRICS_CSS = """
<style>
    /* ── Metric card wrapper ─────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.08), rgba(108, 99, 255, 0.03));
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(108, 99, 255, 0.35);
        box-shadow: 0 4px 24px rgba(108, 99, 255, 0.12);
        transform: translateY(-2px);
    }
    .metric-card .metric-icon {
        font-size: 1.6rem;
        margin-bottom: 0.3rem;
    }
    .metric-card .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.2rem 0;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        color: #7a78a0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-card .metric-delta {
        font-size: 0.72rem;
        margin-top: 0.2rem;
    }
    .metric-card .metric-delta.positive { color: #4ade80; }
    .metric-card .metric-delta.negative { color: #f87171; }
    .metric-card .metric-delta.neutral  { color: #7a78a0; }

    /* ── Document stats row ──────────────────────────────────── */
    .doc-stats-container {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 0.5rem 0;
    }
    .doc-stat-item {
        flex: 1;
        min-width: 120px;
        background: rgba(108, 99, 255, 0.06);
        border: 1px solid rgba(108, 99, 255, 0.12);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        text-align: center;
    }
    .doc-stat-item .doc-stat-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #c9c5ff;
    }
    .doc-stat-item .doc-stat-label {
        font-size: 0.72rem;
        color: #7a78a0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
</style>
"""


def _build_metric_card(
    icon: str,
    label: str,
    value: str,
    delta: str = "",
    delta_type: str = "neutral",
) -> str:
    """
    Return the HTML for a single metric card.

    Args:
        icon: Emoji or icon string.
        label: Short label displayed below the value.
        value: The primary metric value (already formatted).
        delta: Optional delta indicator text.
        delta_type: One of 'positive', 'negative', or 'neutral'.
    """
    delta_html = ""
    if delta:
        delta_html = (
            f'<div class="metric-delta {delta_type}">{delta}</div>'
        )

    return (
        f'<div class="metric-card">'
        f'  <div class="metric-icon">{icon}</div>'
        f'  <div class="metric-value">{value}</div>'
        f'  <div class="metric-label">{label}</div>'
        f"  {delta_html}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_metrics(metrics: Dict[str, Any]) -> None:
    """
    Render a row of four metric cards for the latest query.

    Expected keys in *metrics*:
        - retrieval_time_ms (float)
        - generation_time_ms (float)
        - num_chunks (int)
        - num_sources (int, optional – derived if missing)
    """
    st.markdown(METRICS_CSS, unsafe_allow_html=True)

    retrieval_ms: float = metrics.get("retrieval_time_ms", 0)
    generation_ms: float = metrics.get("generation_time_ms", 0)
    num_chunks: int = metrics.get("num_chunks", 0)
    num_sources: int = metrics.get(
        "num_sources", st.session_state.get("num_documents", 0)
    )

    # Determine delta hints (compare against "acceptable" thresholds)
    retrieval_delta = "Fast ✓" if retrieval_ms < 500 else "Slow ✗"
    retrieval_delta_type = "positive" if retrieval_ms < 500 else "negative"

    generation_delta = "Fast ✓" if generation_ms < 3000 else "Slow ✗"
    generation_delta_type = "positive" if generation_ms < 3000 else "negative"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            _build_metric_card(
                icon="🔍",
                label="Retrieval Latency",
                value=format_time(retrieval_ms),
                delta=retrieval_delta,
                delta_type=retrieval_delta_type,
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            _build_metric_card(
                icon="⚡",
                label="Generation Latency",
                value=format_time(generation_ms),
                delta=generation_delta,
                delta_type=generation_delta_type,
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            _build_metric_card(
                icon="📦",
                label="Chunks Retrieved",
                value=str(num_chunks),
            ),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            _build_metric_card(
                icon="📄",
                label="Source Documents",
                value=str(num_sources),
            ),
            unsafe_allow_html=True,
        )


def render_document_stats() -> None:
    """
    Render overall document-level statistics from session_state.

    Displays number of documents, total chunks, file names, and
    processing status.
    """
    st.markdown(METRICS_CSS, unsafe_allow_html=True)

    if not st.session_state.get("documents_processed"):
        st.info(
            "📭 No documents have been processed yet. "
            "Upload files in the sidebar and click **Process**."
        )
        return

    num_docs: int = st.session_state.get("num_documents", 0)
    num_chunks: int = st.session_state.get("num_chunks", 0)
    file_names: list = st.session_state.get("file_names", [])
    has_faiss: bool = st.session_state.get("faiss_index") is not None
    has_bm25: bool = st.session_state.get("keyword_retriever") is not None

    st.markdown(
        f"""
        <div class="doc-stats-container">
            <div class="doc-stat-item">
                <div class="doc-stat-value">{num_docs}</div>
                <div class="doc-stat-label">Documents</div>
            </div>
            <div class="doc-stat-item">
                <div class="doc-stat-value">{num_chunks}</div>
                <div class="doc-stat-label">Total Chunks</div>
            </div>
            <div class="doc-stat-item">
                <div class="doc-stat-value">{"✅" if has_faiss else "❌"}</div>
                <div class="doc-stat-label">FAISS Index</div>
            </div>
            <div class="doc-stat-item">
                <div class="doc-stat-value">{"✅" if has_bm25 else "❌"}</div>
                <div class="doc-stat-label">BM25 Index</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show file list
    if file_names:
        st.caption("**Loaded files:** " + ", ".join(file_names))
