"""
Chat interface component for the Knowledge Copilot application.

Provides a conversational UI using Streamlit's chat_message API.
Handles hybrid retrieval, answer generation via Groq, citation rendering,
chunk inspection, and chat export.
"""

import time
import streamlit as st
from typing import Dict, List, Any, Optional

# --- Backend imports ---
from retrieval.semantic_retriever import SemanticRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.pageindex_retriever import PageIndexRetriever
from workflow.graph import run_adaptive_rag
from llm.groq_llm import get_groq_llm, generate_answer
from utils.citation_formatter import (
    format_citations,
    format_citations_markdown,
    group_citations_by_source,
)
from utils.helpers import format_time, export_chat_to_markdown


# ---------------------------------------------------------------------------
# Custom CSS specific to the chat area
# ---------------------------------------------------------------------------
CHAT_CSS = """
<style>
    /* ── Chat message containers ─────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(108, 99, 255, 0.08);
        transition: box-shadow 0.3s ease;
    }
    [data-testid="stChatMessage"]:hover {
        box-shadow: 0 2px 16px rgba(108, 99, 255, 0.10);
    }

    /* ── Metrics pills inside chat ───────────────────────────── */
    .chat-metrics-bar {
        display: flex;
        gap: 0.7rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .chat-metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(108, 99, 255, 0.10);
        border: 1px solid rgba(108, 99, 255, 0.20);
        border-radius: 20px;
        padding: 0.22rem 0.7rem;
        font-size: 0.74rem;
        color: #b0adff;
    }

    /* ── Citation card ───────────────────────────────────────── */
    .citation-card {
        background: rgba(108, 99, 255, 0.06);
        border-left: 3px solid #6C63FF;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 0.9rem;
        margin: 0.35rem 0;
        font-size: 0.82rem;
        color: #d0ceff;
    }
    .citation-card .citation-source {
        font-weight: 600;
        color: #9d99ff;
        font-size: 0.78rem;
    }

    /* ── Chunk viewer ────────────────────────────────────────── */
    .chunk-viewer {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(108, 99, 255, 0.12);
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        margin: 0.35rem 0;
        font-size: 0.8rem;
        line-height: 1.5;
    }
    .chunk-viewer .chunk-meta {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.3rem;
        font-size: 0.72rem;
        color: #7a78a0;
    }
    .chunk-viewer .chunk-meta .retriever-badge {
        background: rgba(108, 99, 255, 0.15);
        border-radius: 4px;
        padding: 0.1rem 0.5rem;
        color: #b0adff;
    }

    /* ── Empty-state hero ────────────────────────────────────── */
    .chat-empty-state {
        text-align: center;
        padding: 4rem 1rem 2rem 1rem;
        color: #7a78a0;
    }
    .chat-empty-state h2 {
        background: linear-gradient(135deg, #6C63FF, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.6rem;
        margin-bottom: 0.5rem;
    }
    .chat-empty-state p {
        font-size: 0.9rem;
        max-width: 500px;
        margin: 0 auto;
    }

    /* ── Download button ─────────────────────────────────────── */
    .download-btn-row {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 0.5rem;
    }
</style>
"""


def _build_context_from_history(messages: List[Dict], last_n: int = 5) -> str:
    """
    Build a short textual context string from the most recent *last_n*
    messages so the LLM can handle follow-up questions.
    """
    recent = messages[-last_n:]
    parts: List[str] = []
    for msg in recent:
        role = msg["role"].capitalize()
        parts.append(f"{role}: {msg['content']}")
    return "\n".join(parts)


def _render_citations(citations: List[Dict]) -> None:
    """Render formatted citation cards inside an expander."""
    if not citations:
        st.info("No citations available for this response.")
        return

    grouped = group_citations_by_source(citations)
    for source, items in grouped.items():
        st.markdown(
            f'<div class="citation-card">'
            f'<span class="citation-source">📄 {source}</span><br>'
            + "<br>".join(
                [
                    f"• Page {c.get('page_number', '–')}: {c.get('text', '')[:120]}…"
                    for c in items
                ]
            )
            + "</div>",
            unsafe_allow_html=True,
        )


def _render_chunks(chunks: List[Dict]) -> None:
    """Show each retrieved chunk with metadata inside an expander."""
    if not chunks:
        st.info("No chunks to display.")
        return

    for i, chunk in enumerate(chunks, start=1):
        doc = chunk.get("document", None)
        content = doc.page_content if doc else chunk.get("text", "")
        metadata = doc.metadata if doc else {}
        source = metadata.get("source", "Unknown")
        page = metadata.get("page_number", "–")
        score = chunk.get("score", 0)
        retriever = chunk.get("retriever", chunk.get("retrievers", "–"))

        # Truncate very long chunks for display
        display_content = content[:400] + ("…" if len(content) > 400 else "")

        st.markdown(
            f"""
            <div class="chunk-viewer">
                <div class="chunk-meta">
                    <span>#{i} · {source} · p.{page} · score {score:.3f}</span>
                    <span class="retriever-badge">{retriever}</span>
                </div>
                {display_content}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_metrics_pills(metrics: Dict[str, Any]) -> None:
    """Display tiny metric pills below an assistant message."""
    pills: List[str] = []
    if "strategy" in metrics:
        pills.append(f"🎯 Strategy: {str(metrics['strategy']).upper()}")
    if metrics.get("retries", 0) > 0:
        pills.append(f"🔄 Retries: {metrics['retries']}")
    if "retrieval_time_ms" in metrics and metrics["retrieval_time_ms"] > 0:
        pills.append(f"🔍 Retrieval: {format_time(metrics['retrieval_time_ms'])}")
    if "generation_time_ms" in metrics and metrics["generation_time_ms"] > 0:
        pills.append(f"⚡ Generation: {format_time(metrics['generation_time_ms'])}")
    if "total_time_ms" in metrics:
        pills.append(f"⏱️ Total: {format_time(metrics['total_time_ms'])}")
    if "num_chunks" in metrics:
        pills.append(f"📦 Chunks: {metrics['num_chunks']}")

    if pills:
        html = "".join(
            [f'<span class="chat-metric-pill">{p}</span>' for p in pills]
        )
        st.markdown(
            f'<div class="chat-metrics-bar">{html}</div>', unsafe_allow_html=True
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_chat_interface(settings: Dict[str, Any]) -> None:
    """
    Render the main chat interface.

    Args:
        settings: Dict returned by ``sidebar.render_sidebar()``.
    """
    # Inject CSS
    st.markdown(CHAT_CSS, unsafe_allow_html=True)

    # Ensure chat history exists
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    messages: List[Dict] = st.session_state["messages"]

    # ── Download chat button ────────────────────────────────────
    if messages:
        col_spacer, col_dl = st.columns([5, 1])
        with col_dl:
            md_export = export_chat_to_markdown(messages)
            st.download_button(
                label="💾 Download Chat",
                data=md_export,
                file_name="knowledge_copilot_chat.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # ── Render existing messages ────────────────────────────────
    if not messages:
        st.markdown(
            """
            <div class="chat-empty-state">
                <h2>Ask anything about your documents</h2>
                <p>
                    Upload documents in the sidebar, click <strong>Process</strong>,
                    then ask questions here. The hybrid retrieval engine will search
                    using both semantic and keyword matching for the best answers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # For assistant messages, show extras
            if msg["role"] == "assistant":
                # Metrics pills
                if msg.get("metrics"):
                    _render_metrics_pills(msg["metrics"])
                # Adaptive reasoning trace
                if msg.get("metrics", {}).get("diagnostic_trace"):
                    with st.expander("🧠 Adaptive Routing & Reasoning Trace", expanded=False):
                        for step in msg["metrics"]["diagnostic_trace"]:
                            st.markdown(f"- {step}")
                # Citations
                if msg.get("citations"):
                    with st.expander("📋 Citations", expanded=False):
                        _render_citations(msg["citations"])
                # Chunks
                if msg.get("chunks"):
                    with st.expander("📦 Retrieved Chunks", expanded=False):
                        _render_chunks(msg["chunks"])

    # ── Chat input ──────────────────────────────────────────────
    user_input: Optional[str] = st.chat_input(
        "Ask a question about your documents…"
    )

    if user_input:
        _handle_user_message(user_input, settings)


def _handle_user_message(query: str, settings: Dict[str, Any]) -> None:
    """
    Process a new user query through the full RAG pipeline:
      retrieve ➜ generate ➜ display.
    """
    messages: List[Dict] = st.session_state["messages"]

    # ── Display user message ────────────────────────────────────
    messages.append(
        {
            "role": "user",
            "content": query,
            "citations": [],
            "chunks": [],
            "metrics": {},
        }
    )
    with st.chat_message("user"):
        st.markdown(query)

    # ── Validate prerequisites ──────────────────────────────────
    api_key: str = settings.get("api_key", "")
    if not api_key:
        with st.chat_message("assistant"):
            st.warning(
                "⚠️ **Groq API key is missing.** Please enter your API key in "
                "the sidebar under *Model Settings* to enable answer generation."
            )
        messages.append(
            {
                "role": "assistant",
                "content": "⚠️ Groq API key is missing. Please provide it in the sidebar.",
                "citations": [],
                "chunks": [],
                "metrics": {},
            }
        )
        return

    if not st.session_state.get("documents_processed"):
        with st.chat_message("assistant"):
            st.warning(
                "📄 **No documents processed yet.** Upload files in the sidebar "
                "and click **Process** before asking questions."
            )
        messages.append(
            {
                "role": "assistant",
                "content": "📄 No documents processed yet. Upload and process documents first.",
                "citations": [],
                "chunks": [],
                "metrics": {},
            }
        )
        return

    # ── Run adaptive retrieval & generation workflow ────────────
    with st.chat_message("assistant"):
        with st.spinner("🧠 RAGNEXUS Adaptive Reasoning & Retrieval…"):
            start_total = time.time()
            try:
                faiss_idx = st.session_state.get("faiss_index")
                semantic_ret = SemanticRetriever(faiss_idx) if faiss_idx else None
                keyword_ret = st.session_state.get("keyword_retriever")
                pageindex_ret = st.session_state.get("pageindex_retriever")
                hybrid_ret = (
                    HybridRetriever(
                        semantic_retriever=semantic_ret,
                        keyword_retriever=keyword_ret,
                    )
                    if semantic_ret and keyword_ret
                    else None
                )

                llm = get_groq_llm(
                    model_name=settings.get("model_name", ""),
                    api_key=api_key,
                )

                final_state = run_adaptive_rag(
                    query=query,
                    semantic_retriever=semantic_ret,
                    keyword_retriever=keyword_ret,
                    pageindex_retriever=pageindex_ret,
                    hybrid_retriever=hybrid_ret,
                    llm=llm,
                    max_retries=2,
                )

                answer = final_state.get("answer", "")
                retrieved_chunks = final_state.get("retrieved_chunks", [])
                citations = final_state.get("citations", [])
                strategy = final_state.get("strategy", "hybrid")
                retries = final_state.get("retry_count", 0)
                diagnostic_trace = final_state.get("diagnostic_trace", [])

            except Exception as exc:
                err_msg = str(exc)
                if "model_not_found" in err_msg or "404" in err_msg:
                    help_note = (
                        f"❌ **Model Error:** The model `{settings.get('model_name')}` is not accessible with your Groq API key.\n\n"
                        f"👉 **Fix:** Check **Model Settings** in the sidebar and choose an active model supported by your account (such as **GPT-OSS 120B** or **Qwen 3.8 27B**)."
                    )
                    st.error(help_note)
                    msg_content = help_note
                else:
                    st.error(f"Workflow execution failed: {exc}")
                    msg_content = f"❌ Workflow execution failed: {exc}"

                messages.append(
                    {
                        "role": "assistant",
                        "content": msg_content,
                        "citations": [],
                        "chunks": [],
                        "metrics": {},
                    }
                )
                return

        end_total = time.time()
        total_time = (end_total - start_total) * 1000  # ms

        # ── Assemble metrics ────────────────────────────────────
        metrics: Dict[str, Any] = {
            "strategy": strategy,
            "retries": retries,
            "total_time_ms": total_time,
            "num_chunks": len(retrieved_chunks),
            "diagnostic_trace": diagnostic_trace,
        }

        # ── Display answer ──────────────────────────────────────
        st.markdown(answer)

        # Metrics pills
        _render_metrics_pills(metrics)

        # Adaptive trace expander
        if diagnostic_trace:
            with st.expander("🧠 Adaptive Routing & Reasoning Trace", expanded=False):
                for step in diagnostic_trace:
                    st.markdown(f"- {step}")

        # Citations expander
        if citations:
            with st.expander("📋 Citations", expanded=False):
                _render_citations(citations)

        # Chunks expander
        if retrieved_chunks:
            with st.expander("📦 Retrieved Chunks", expanded=False):
                _render_chunks(retrieved_chunks)

        # ── Save to history ─────────────────────────────────────
        messages.append(
            {
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "chunks": retrieved_chunks,
                "metrics": metrics,
            }
        )

