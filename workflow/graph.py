"""Adaptive RAG StateGraph assembly using LangGraph."""

from __future__ import annotations

from typing import Any, Dict, Literal
from langgraph.graph import StateGraph, START, END

from workflow.state import RAGGraphState
from workflow.nodes import (
    route_node,
    retrieve_node,
    grade_context_node,
    rewrite_query_node,
    generate_node,
    verify_grounding_node,
    fallback_node,
)


def decide_after_grading(
    state: RAGGraphState,
) -> Literal["generate", "rewrite_query", "fallback"]:
    """Conditional transition routing based on context grading and retry limits."""
    grade = state.get("context_grade", "inadequate")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if grade == "relevant":
        return "generate"
    elif retry_count < max_retries:
        return "rewrite_query"
    else:
        return "fallback"


def build_adaptive_rag_graph():
    """Build and compile the adaptive stateful LangGraph for RAGNEXUS."""
    workflow = StateGraph(RAGGraphState)

    # Add all workflow nodes
    workflow.add_node("route", route_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_context", grade_context_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify_grounding", verify_grounding_node)
    workflow.add_node("fallback", fallback_node)

    # Connect workflow edges
    workflow.add_edge(START, "route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "grade_context")

    # Conditional branching from grade_context
    workflow.add_conditional_edges(
        "grade_context",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )

    # Loop back from rewrite_query to route
    workflow.add_edge("rewrite_query", "route")

    # Generation and verification
    workflow.add_edge("generate", "verify_grounding")
    workflow.add_edge("verify_grounding", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


def run_adaptive_rag(
    query: str,
    semantic_retriever: Any = None,
    keyword_retriever: Any = None,
    pageindex_retriever: Any = None,
    hybrid_retriever: Any = None,
    llm: Any = None,
    max_retries: int = 2,
) -> RAGGraphState:
    """Convenience runner for executing the compiled adaptive graph.

    Args:
        query: User input query.
        semantic_retriever: SemanticRetriever instance.
        keyword_retriever: KeywordRetriever instance.
        pageindex_retriever: PageIndexRetriever instance.
        hybrid_retriever: HybridRetriever instance.
        llm: ChatGroq or compatible LLM instance.
        max_retries: Maximum permitted retries (default 2).

    Returns:
        Final RAGGraphState containing answer, citations, strategy, and diagnostics.
    """
    graph = build_adaptive_rag_graph()

    initial_state: RAGGraphState = {
        "query": query,
        "original_query": query,
        "query_history": [],
        "retry_count": 0,
        "max_retries": max_retries,
        "diagnostic_trace": [f"Initiated adaptive RAG for query: '{query}'"],
        "semantic_retriever": semantic_retriever,
        "keyword_retriever": keyword_retriever,
        "pageindex_retriever": pageindex_retriever,
        "hybrid_retriever": hybrid_retriever,
        "llm": llm,
    }

    final_state = graph.invoke(initial_state)
    return final_state
