"""State definition for RAGNEXUS Adaptive LangGraph workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class RAGGraphState(TypedDict, total=False):
    """Explicit state passing through the adaptive RAG workflow graph."""

    # Query inputs & tracking
    query: str
    original_query: str
    query_history: List[str]

    # Strategy & routing
    strategy: str  # "vector" | "bm25" | "pageindex" | "hybrid"
    routing_reason: str
    routing_confidence: float

    # Retrieval
    retrieved_chunks: List[Dict[str, Any]]
    context_text: str

    # Context Grading & Retries
    context_grade: str  # "relevant" | "inadequate"
    grade_reason: str
    retry_count: int
    max_retries: int

    # Generation & Grounding
    answer: str
    citations: List[Dict[str, Any]]
    is_grounded: bool
    diagnostic_trace: List[str]

    # Shared runtime references
    semantic_retriever: Any
    keyword_retriever: Any
    pageindex_retriever: Any
    hybrid_retriever: Any
    llm: Any
