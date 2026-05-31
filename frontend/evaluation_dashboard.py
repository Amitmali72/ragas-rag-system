"""
Evaluation dashboard component for the Knowledge Copilot application.

Provides two tabbed views:
  1. **RAGAS Evaluation** – Run standardised RAG quality metrics on a set
     of question/ground-truth pairs and visualise results.
  2. **LLM vs RAG Comparison** – Side-by-side comparison of direct LLM
     answers versus Hybrid RAG answers with metric overlays.

Uses Plotly for charts and Streamlit's native widgets for layout.
"""

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional

# --- Backend imports ---
from retrieval.hybrid_retriever import HybridRetriever
from llm.groq_llm import get_groq_llm, generate_answer, generate_direct_answer
from evaluation.ragas_evaluator import evaluate_rag
from evaluation.comparison import run_comparison
from utils.helpers import format_time


# ---------------------------------------------------------------------------
# Custom CSS for the dashboard
# ---------------------------------------------------------------------------
EVAL_CSS = """
<style>
    /* ── Tab panel ───────────────────────────────────────────── */
    .eval-tab-panel {
        padding: 0.5rem 0;
    }

    /* ── Score badge ─────────────────────────────────────────── */
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .score-badge.good   { background: rgba(74,222,128,0.15); color: #4ade80; }
    .score-badge.medium { background: rgba(250,204,21,0.15); color: #facc15; }
    .score-badge.poor   { background: rgba(248,113,113,0.15); color: #f87171; }

    /* ── Comparison card ─────────────────────────────────────── */
    .comparison-answer-card {
        background: rgba(108, 99, 255, 0.05);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        min-height: 160px;
    }
    .comparison-answer-card h4 {
        margin: 0 0 0.5rem 0;
        color: #9d99ff;
        font-size: 0.9rem;
    }

    /* ── Improvement badge ───────────────────────────────────── */
    .improvement-positive { color: #4ade80; font-weight: 600; }
    .improvement-negative { color: #f87171; font-weight: 600; }
</style>
"""

# Sample evaluation data for quick testing
SAMPLE_EVAL_DATA: List[Dict[str, str]] = [
    {
        "question": "What are the main components of a Retrieval Augmented Generation system?",
        "ground_truth": "A RAG system consists of a retriever that fetches relevant documents from a knowledge base, and a generator (typically a large language model) that produces answers conditioned on both the query and the retrieved context.",
    },
    {
        "question": "How does BM25 keyword search work?",
        "ground_truth": "BM25 is a probabilistic ranking function that scores documents based on term frequency, inverse document frequency, and document length normalization. It extends TF-IDF with saturation of term frequency and document length penalization.",
    },
    {
        "question": "What is the purpose of text chunking in RAG pipelines?",
        "ground_truth": "Text chunking splits large documents into smaller, manageable pieces that can be individually embedded and retrieved. This ensures the retriever can find the most relevant portion of a document rather than returning entire documents.",
    },
]


def _score_badge(score: float) -> str:
    """Return an HTML score badge with colour based on threshold."""
    if score >= 0.8:
        cls = "good"
    elif score >= 0.5:
        cls = "medium"
    else:
        cls = "poor"
    return f'<span class="score-badge {cls}">{score:.2f}</span>'


def _create_radar_chart(
    metrics: Dict[str, float],
    title: str = "RAGAS Metrics",
    color: str = "rgba(108, 99, 255, 0.7)",
    fill_color: str = "rgba(108, 99, 255, 0.15)",
) -> go.Figure:
    """
    Create a Plotly radar (polar) chart for a set of named metrics.

    Args:
        metrics: Mapping of metric-name → score (0-1).
        title: Chart title.
        color: Line colour.
        fill_color: Fill colour.
    """
    categories = list(metrics.keys())
    values = list(metrics.values())
    # Close the polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=color, width=2),
            name=title,
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1], gridcolor="rgba(108,99,255,0.1)"
            ),
            angularaxis=dict(gridcolor="rgba(108,99,255,0.1)"),
        ),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=60, r=60, t=60, b=40),
        height=380,
    )
    return fig


def _create_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    color: str = "#6C63FF",
) -> go.Figure:
    """Create a styled Plotly bar chart."""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        xaxis=dict(gridcolor="rgba(108,99,255,0.08)"),
        yaxis=dict(gridcolor="rgba(108,99,255,0.08)", range=[0, 1]),
        margin=dict(l=40, r=20, t=50, b=40),
        height=350,
    )
    return fig


def _create_grouped_bar_chart(
    df: pd.DataFrame,
    title: str = "",
) -> go.Figure:
    """Create a grouped bar chart comparing two methods across metrics."""
    fig = go.Figure()

    if "Direct LLM" in df.columns:
        fig.add_trace(
            go.Bar(
                name="Direct LLM",
                x=df["Metric"],
                y=df["Direct LLM"],
                marker_color="rgba(248, 113, 113, 0.7)",
            )
        )
    if "Hybrid RAG" in df.columns:
        fig.add_trace(
            go.Bar(
                name="Hybrid RAG",
                x=df["Metric"],
                y=df["Hybrid RAG"],
                marker_color="rgba(108, 99, 255, 0.7)",
            )
        )

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        yaxis=dict(range=[0, 1], gridcolor="rgba(108,99,255,0.08)"),
        xaxis=dict(gridcolor="rgba(108,99,255,0.08)"),
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=40, r=20, t=50, b=40),
        height=380,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def _render_ragas_tab(settings: Dict[str, Any]) -> None:
    """Render the RAGAS Evaluation tab contents."""
    st.markdown("#### 📊 RAGAS Evaluation Suite")
    st.caption(
        "Upload a JSON file with question/ground_truth pairs, or use the "
        "built-in sample data to evaluate your RAG pipeline."
    )

    # ── Input selection ─────────────────────────────────────────
    col_upload, col_sample = st.columns([3, 1])
    eval_data: Optional[List[Dict[str, str]]] = None

    with col_upload:
        uploaded = st.file_uploader(
            "Upload evaluation JSON",
            type=["json"],
            key="ragas_upload",
            help='Format: [{"question": "...", "ground_truth": "..."}]',
        )
        if uploaded:
            try:
                eval_data = json.load(uploaded)
                st.success(f"Loaded **{len(eval_data)}** evaluation pairs.")
            except Exception as exc:
                st.error(f"Invalid JSON: {exc}")

    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📋 Use Samples", use_container_width=True):
            eval_data = SAMPLE_EVAL_DATA
            st.info(f"Using **{len(SAMPLE_EVAL_DATA)}** sample questions.")

    # Persist eval data in session state so it survives button clicks
    if eval_data is not None:
        st.session_state["eval_data"] = eval_data
    eval_data = st.session_state.get("eval_data")

    if not eval_data:
        return

    # Preview the data
    with st.expander("👀 Preview evaluation data", expanded=False):
        st.dataframe(
            pd.DataFrame(eval_data),
            use_container_width=True,
            hide_index=True,
        )

    # ── Run evaluation ──────────────────────────────────────────
    if not st.button("🚀 Run RAGAS Evaluation", type="primary", key="run_ragas"):
        # Show previous results if available
        if "ragas_results" in st.session_state:
            _display_ragas_results(st.session_state["ragas_results"])
        return

    # Pre-flight checks
    if not st.session_state.get("documents_processed"):
        st.warning("⚠️ Process documents first before running evaluation.")
        return
    api_key = settings.get("api_key", "")
    if not api_key:
        st.warning("⚠️ Please set a Groq API key in the sidebar.")
        return

    # Run pipeline for each question
    progress = st.progress(0, text="Running evaluation…")
    results: List[Dict[str, Any]] = []
    total = len(eval_data)

    hybrid_retriever = HybridRetriever(
        faiss_index=st.session_state["faiss_index"],
        keyword_retriever=st.session_state["keyword_retriever"],
    )
    llm = get_groq_llm(model_name=settings["model_name"], api_key=api_key)
    top_k = settings.get("top_k", 5)

    for i, item in enumerate(eval_data):
        progress.progress(
            int((i / total) * 80),
            text=f"Evaluating question {i + 1}/{total}…",
        )
        question = item["question"]
        ground_truth = item["ground_truth"]

        try:
            # Retrieve
            chunks, _ = hybrid_retriever.retrieve(question, top_k, top_k)
            context_docs = [c["document"] for c in chunks if c.get("document")]

            # Generate
            answer, gen_time = generate_answer(question, context_docs, llm)

            results.append(
                {
                    "question": question,
                    "answer": answer,
                    "ground_truth": ground_truth,
                    "contexts": [d.page_content for d in context_docs],
                }
            )
        except Exception as exc:
            st.error(f"Error on Q{i + 1}: {exc}")

    # Run RAGAS evaluation
    progress.progress(85, text="Computing RAGAS metrics…")
    try:
        ragas_results = evaluate_rag(results)
    except Exception as exc:
        progress.empty()
        st.error(f"RAGAS evaluation failed: {exc}")
        return

    progress.progress(100, text="✅ Evaluation complete!")
    st.session_state["ragas_results"] = ragas_results
    _display_ragas_results(ragas_results)


def _display_ragas_results(ragas_results: Any) -> None:
    """Render RAGAS evaluation results with charts and tables."""
    st.markdown("---")
    st.markdown("#### 📈 Evaluation Results")

    # Extract aggregate metrics (handle both dict and object forms)
    if isinstance(ragas_results, dict):
        aggregate = ragas_results.get("aggregate", ragas_results)
        per_question = ragas_results.get("per_question", [])
    else:
        aggregate = getattr(ragas_results, "aggregate", {})
        per_question = getattr(ragas_results, "per_question", [])

    # ── Summary metric cards ────────────────────────────────────
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    cols = st.columns(len(metric_names))
    for col, name in zip(cols, metric_names):
        val = aggregate.get(name, 0) if isinstance(aggregate, dict) else 0
        with col:
            st.metric(
                label=name.replace("_", " ").title(),
                value=f"{val:.3f}",
            )

    # ── Radar chart ─────────────────────────────────────────────
    radar_metrics = {
        name.replace("_", " ").title(): (
            aggregate.get(name, 0) if isinstance(aggregate, dict) else 0
        )
        for name in metric_names
    }
    fig_radar = _create_radar_chart(radar_metrics, title="RAGAS Aggregate Metrics")
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Per-question table ──────────────────────────────────────
    if per_question:
        st.markdown("##### Per-Question Scores")
        df_per_q = pd.DataFrame(per_question)
        st.dataframe(df_per_q, use_container_width=True, hide_index=True)

        # ── Per-question faithfulness bar chart ─────────────────
        if "faithfulness" in df_per_q.columns:
            df_bar = df_per_q[["question", "faithfulness"]].copy()
            df_bar["question"] = df_bar["question"].str[:60] + "…"
            fig_bar = _create_bar_chart(
                df_bar,
                x_col="question",
                y_col="faithfulness",
                title="Per-Question Faithfulness",
            )
            st.plotly_chart(fig_bar, use_container_width=True)


def _render_comparison_tab(settings: Dict[str, Any]) -> None:
    """Render the LLM vs RAG Comparison tab contents."""
    st.markdown("#### ⚔️ Direct LLM vs Hybrid RAG Comparison")
    st.caption(
        "Compare answers generated by the LLM alone versus the full Hybrid "
        "RAG pipeline. Evaluate which approach produces more accurate, "
        "faithful responses."
    )

    # Reuse the same eval data input
    eval_data = st.session_state.get("eval_data")
    if not eval_data:
        st.info(
            "ℹ️ Go to the **RAGAS Evaluation** tab first and load evaluation "
            "data (upload or samples)."
        )
        return

    st.caption(f"Using **{len(eval_data)}** evaluation questions.")

    if not st.button("⚔️ Run Comparison", type="primary", key="run_comparison"):
        if "comparison_results" in st.session_state:
            _display_comparison_results(st.session_state["comparison_results"])
        return

    # Pre-flight
    if not st.session_state.get("documents_processed"):
        st.warning("⚠️ Process documents first.")
        return
    api_key = settings.get("api_key", "")
    if not api_key:
        st.warning("⚠️ Set a Groq API key in the sidebar.")
        return

    progress = st.progress(0, text="Running comparison…")
    llm = get_groq_llm(model_name=settings["model_name"], api_key=api_key)
    hybrid_retriever = HybridRetriever(
        faiss_index=st.session_state["faiss_index"],
        keyword_retriever=st.session_state["keyword_retriever"],
    )
    top_k = settings.get("top_k", 5)

    comparison_items: List[Dict[str, Any]] = []
    total = len(eval_data)

    for i, item in enumerate(eval_data):
        progress.progress(
            int((i / total) * 80),
            text=f"Comparing question {i + 1}/{total}…",
        )
        question = item["question"]
        ground_truth = item["ground_truth"]

        try:
            # ── Direct LLM ──────────────────────────────────────
            direct_answer, direct_time = generate_direct_answer(question, llm)

            # ── Hybrid RAG ──────────────────────────────────────
            chunks, _ = hybrid_retriever.retrieve(question, top_k, top_k)
            context_docs = [c["document"] for c in chunks if c.get("document")]
            rag_answer, rag_time = generate_answer(question, context_docs, llm)

            comparison_items.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "direct_answer": direct_answer,
                    "rag_answer": rag_answer,
                    "direct_time_ms": direct_time,
                    "rag_time_ms": rag_time,
                    "contexts": [d.page_content for d in context_docs],
                }
            )
        except Exception as exc:
            st.error(f"Error on Q{i + 1}: {exc}")

    # Run RAGAS on both sets
    progress.progress(85, text="Computing comparison metrics…")
    try:
        comparison_results = run_comparison(comparison_items)
    except Exception as exc:
        progress.empty()
        st.error(f"Comparison evaluation failed: {exc}")
        return

    progress.progress(100, text="✅ Comparison complete!")
    st.session_state["comparison_results"] = comparison_results
    _display_comparison_results(comparison_results)


def _display_comparison_results(results: Any) -> None:
    """Render comparison results with side-by-side answers and charts."""
    st.markdown("---")

    # Extract structured data
    if isinstance(results, dict):
        items = results.get("items", [])
        summary = results.get("summary", {})
    else:
        items = getattr(results, "items", [])
        summary = getattr(results, "summary", {})

    # ── Side-by-side answers ────────────────────────────────────
    st.markdown("##### 📝 Side-by-Side Answers")
    for idx, item in enumerate(items):
        question = item.get("question", f"Question {idx + 1}")
        with st.expander(f"Q{idx + 1}: {question[:80]}…", expanded=(idx == 0)):
            col_direct, col_rag = st.columns(2)
            with col_direct:
                st.markdown(
                    f'<div class="comparison-answer-card">'
                    f"<h4>🤖 Direct LLM</h4>"
                    f"{item.get('direct_answer', 'N/A')}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_rag:
                st.markdown(
                    f'<div class="comparison-answer-card">'
                    f"<h4>🧠 Hybrid RAG</h4>"
                    f"{item.get('rag_answer', 'N/A')}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Grouped bar chart of RAGAS metrics ──────────────────────
    if isinstance(summary, dict) and summary:
        st.markdown("##### 📊 Metric Comparison")
        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        chart_data: Dict[str, list] = {"Metric": [], "Direct LLM": [], "Hybrid RAG": []}

        direct_metrics = summary.get("direct_llm", {})
        rag_metrics = summary.get("hybrid_rag", {})

        for m in metric_names:
            chart_data["Metric"].append(m.replace("_", " ").title())
            chart_data["Direct LLM"].append(
                direct_metrics.get(m, 0) if isinstance(direct_metrics, dict) else 0
            )
            chart_data["Hybrid RAG"].append(
                rag_metrics.get(m, 0) if isinstance(rag_metrics, dict) else 0
            )

        df_chart = pd.DataFrame(chart_data)
        fig_grouped = _create_grouped_bar_chart(
            df_chart, title="Direct LLM vs Hybrid RAG – RAGAS Metrics"
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

        # ── Radar overlay ───────────────────────────────────────
        fig_overlay = go.Figure()

        direct_vals = [direct_metrics.get(m, 0) for m in metric_names]
        rag_vals = [rag_metrics.get(m, 0) for m in metric_names]
        theta_labels = [m.replace("_", " ").title() for m in metric_names]

        # Close polygons
        direct_vals.append(direct_vals[0])
        rag_vals.append(rag_vals[0])
        theta_labels_closed = theta_labels + [theta_labels[0]]

        fig_overlay.add_trace(
            go.Scatterpolar(
                r=direct_vals,
                theta=theta_labels_closed,
                fill="toself",
                fillcolor="rgba(248,113,113,0.12)",
                line=dict(color="rgba(248,113,113,0.7)", width=2),
                name="Direct LLM",
            )
        )
        fig_overlay.add_trace(
            go.Scatterpolar(
                r=rag_vals,
                theta=theta_labels_closed,
                fill="toself",
                fillcolor="rgba(108,99,255,0.12)",
                line=dict(color="rgba(108,99,255,0.7)", width=2),
                name="Hybrid RAG",
            )
        )

        fig_overlay.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor="rgba(108,99,255,0.1)",
                ),
                angularaxis=dict(gridcolor="rgba(108,99,255,0.1)"),
            ),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            title=dict(text="Radar Overlay – LLM vs RAG", font=dict(size=14)),
            margin=dict(l=60, r=60, t=60, b=40),
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        st.plotly_chart(fig_overlay, use_container_width=True)

        # ── Summary table with improvement % ────────────────────
        st.markdown("##### 📋 Improvement Summary")
        summary_rows: List[Dict[str, Any]] = []
        for m in metric_names:
            d_val = direct_metrics.get(m, 0) if isinstance(direct_metrics, dict) else 0
            r_val = rag_metrics.get(m, 0) if isinstance(rag_metrics, dict) else 0
            improvement = ((r_val - d_val) / d_val * 100) if d_val != 0 else 0
            summary_rows.append(
                {
                    "Metric": m.replace("_", " ").title(),
                    "Direct LLM": f"{d_val:.3f}",
                    "Hybrid RAG": f"{r_val:.3f}",
                    "Improvement": f"{improvement:+.1f}%",
                }
            )
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_evaluation_dashboard(settings: Dict[str, Any]) -> None:
    """
    Render the full evaluation dashboard with RAGAS and comparison tabs.

    Args:
        settings: Dict returned by ``sidebar.render_sidebar()``.
    """
    st.markdown(EVAL_CSS, unsafe_allow_html=True)

    tab_ragas, tab_compare = st.tabs(
        ["📊 RAGAS Evaluation", "⚔️ LLM vs RAG Comparison"]
    )

    with tab_ragas:
        _render_ragas_tab(settings)

    with tab_compare:
        _render_comparison_tab(settings)
