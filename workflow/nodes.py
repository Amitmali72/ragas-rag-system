"""Workflow nodes for the RAGNEXUS Adaptive LangGraph pipeline."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document

from workflow.state import RAGGraphState
from workflow.router import route_query_strategy
from utils.citation_formatter import format_citations
from llm.groq_llm import format_context, generate_answer


def route_node(state: RAGGraphState) -> RAGGraphState:
    """Analyze query characteristics and select retrieval strategy."""
    query = state.get("query", "")
    strategy, reason, conf = route_query_strategy(query)

    trace = list(state.get("diagnostic_trace", []))
    trace.append(f"Routing: Strategy selected '{strategy}' (confidence: {conf:.2f}). Reason: {reason}")

    return {
        **state,
        "strategy": strategy,
        "routing_reason": reason,
        "routing_confidence": conf,
        "diagnostic_trace": trace,
    }


def retrieve_node(state: RAGGraphState) -> RAGGraphState:
    """Invoke the chosen retriever based on state strategy."""
    strategy = state.get("strategy", "hybrid")
    query = state.get("query", "")
    trace = list(state.get("diagnostic_trace", []))

    retrieved_chunks: List[Dict[str, Any]] = []

    if strategy == "pageindex" and state.get("pageindex_retriever"):
        retriever = state["pageindex_retriever"]
        retrieved_chunks = retriever.retrieve(query, top_k=5)
        trace.append(f"Retrieved {len(retrieved_chunks)} chunks via PageIndex")

    elif strategy == "bm25" and state.get("keyword_retriever"):
        retriever = state["keyword_retriever"]
        retrieved_chunks = retriever.retrieve(query, top_k=5)
        trace.append(f"Retrieved {len(retrieved_chunks)} chunks via BM25 Keyword Search")

    elif strategy == "vector" and state.get("semantic_retriever"):
        retriever = state["semantic_retriever"]
        retrieved_chunks = retriever.retrieve(query, top_k=5)
        trace.append(f"Retrieved {len(retrieved_chunks)} chunks via Vector Semantic Search")

    elif state.get("hybrid_retriever"):
        retriever = state["hybrid_retriever"]
        raw_ret = retriever.retrieve(query, top_k_final=5)
        if isinstance(raw_ret, tuple):
            results, meta = raw_ret
        else:
            results, meta = raw_ret, {}
        retrieved_chunks = results
        time_info = f" (Time: {meta.get('retrieval_time_ms', 0)}ms)" if meta else ""
        trace.append(f"Retrieved {len(retrieved_chunks)} chunks via Hybrid RRF{time_info}")

    elif state.get("semantic_retriever"):
        retriever = state["semantic_retriever"]
        retrieved_chunks = retriever.retrieve(query, top_k=5)
        trace.append(f"Retriever fallback: {len(retrieved_chunks)} chunks via Semantic Search")

    # Format context string
    context_docs = [c["document"] for c in retrieved_chunks if "document" in c]
    context_text = format_context(context_docs)

    return {
        **state,
        "retrieved_chunks": retrieved_chunks,
        "context_text": context_text,
        "diagnostic_trace": trace,
    }


def grade_context_node(state: RAGGraphState) -> RAGGraphState:
    """Grade whether retrieved chunks contain sufficient topical information."""
    query = state.get("query", "").lower()
    chunks = state.get("retrieved_chunks", [])
    trace = list(state.get("diagnostic_trace", []))

    if not chunks:
        trace.append("Context grading: Inadequate (0 chunks retrieved)")
        return {
            **state,
            "context_grade": "inadequate",
            "grade_reason": "No chunks were retrieved for the query.",
            "diagnostic_trace": trace,
        }

    # Extract non-stopword query keywords
    stopwords = {"what", "is", "the", "in", "on", "a", "an", "for", "to", "of", "and", "how", "tell", "me", "about", "show"}
    query_tokens = [w for w in re.findall(r"\w+", query) if w not in stopwords and len(w) > 2]

    combined_text = " ".join(c["document"].page_content.lower() for c in chunks if "document" in c)
    matched_tokens = [t for t in query_tokens if t in combined_text]

    overlap_ratio = (len(matched_tokens) / len(query_tokens)) if query_tokens else 1.0

    if overlap_ratio >= 0.3 or not query_tokens:
        grade = "relevant"
        reason = f"Context satisfies relevance criteria (keyword overlap: {overlap_ratio:.1%})."
    else:
        grade = "inadequate"
        reason = f"Context lacks key query terms (keyword overlap only {overlap_ratio:.1%})."

    trace.append(f"Context grading: {grade.upper()}. {reason}")

    return {
        **state,
        "context_grade": grade,
        "grade_reason": reason,
        "diagnostic_trace": trace,
    }


def rewrite_query_node(state: RAGGraphState) -> RAGGraphState:
    """Rewrite the query to improve retrieval on the next iteration."""
    query = state.get("query", "")
    retry_count = state.get("retry_count", 0) + 1
    history = list(state.get("query_history", []))
    history.append(query)
    trace = list(state.get("diagnostic_trace", []))

    # Strip conversational filler to focus on core terms
    filler_patterns = [
        r"^(?:please\s+)?(?:tell\s+me|explain|what\s+is|show\s+me|can\s+you\s+find)\s+(?:about\s+)?",
        r"\b(?:in\s+detail|briefly|clearly)\b",
    ]
    rewritten = query
    for pat in filler_patterns:
        rewritten = re.sub(pat, "", rewritten, flags=re.IGNORECASE).strip()

    if not rewritten or rewritten == query:
        # Fallback keyword extraction
        words = re.findall(r"\w+", query)
        rewritten = " ".join(w for w in words if len(w) > 3)

    trace.append(f"Query rewrite [Retry {retry_count}]: '{query}' -> '{rewritten}'")

    return {
        **state,
        "query": rewritten,
        "retry_count": retry_count,
        "query_history": history,
        "diagnostic_trace": trace,
    }


def generate_node(state: RAGGraphState) -> RAGGraphState:
    """Generate answer grounded in retrieved context using LLM or structured formatter."""
    query = state.get("original_query", state.get("query", ""))
    chunks = state.get("retrieved_chunks", [])
    llm = state.get("llm")
    trace = list(state.get("diagnostic_trace", []))

    context_docs = [c["document"] for c in chunks if "document" in c]

    if llm is not None:
        try:
            answer, gen_time = generate_answer(query, context_docs, llm)
            trace.append(f"LLM Generation completed in {gen_time:.2f}ms")
        except Exception as exc:
            answer = f"Error during generation: {exc}"
            trace.append(f"LLM Generation error: {exc}")
    else:
        # Offline or test fallback without live LLM
        answer = f"Answer synthesized from {len(context_docs)} document chunks."
        trace.append("Generation synthesized in offline mode")

    citations = format_citations(chunks)

    return {
        **state,
        "answer": answer,
        "citations": citations,
        "diagnostic_trace": trace,
    }


def verify_grounding_node(state: RAGGraphState) -> RAGGraphState:
    """Verify answer citations and ensure strict context grounding."""
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    trace = list(state.get("diagnostic_trace", []))

    if not citations and "not found" not in answer.lower():
        is_grounded = False
        trace.append("Grounding check: Warning - answer generated without citations")
    else:
        is_grounded = True
        trace.append(f"Grounding check: Verified with {len(citations)} citations")

    return {
        **state,
        "is_grounded": is_grounded,
        "diagnostic_trace": trace,
    }


def fallback_node(state: RAGGraphState) -> RAGGraphState:
    """Explicit fallback when retrieval cannot find grounding context after retries."""
    trace = list(state.get("diagnostic_trace", []))
    trace.append(f"Fallback triggered: Maximum retries ({state.get('retry_count', 0)}) reached without sufficient context.")

    return {
        **state,
        "answer": "Information not found in uploaded documents.",
        "citations": [],
        "is_grounded": True,
        "diagnostic_trace": trace,
    }
