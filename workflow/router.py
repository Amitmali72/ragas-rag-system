"""Router module for adaptive retrieval strategy selection.

Analyzes query signals:
  1. Structural cues (page references, section titles, document layout) -> 'pageindex'
  2. Exact identifiers (quotes, technical IDs, codes, specific keywords) -> 'bm25'
  3. Conceptual questions (explanations, why/how, abstract concepts) -> 'vector'
  4. Multi-faceted or default queries -> 'hybrid'
"""

from __future__ import annotations

import re
from typing import Tuple


def route_query_strategy(query: str) -> Tuple[str, str, float]:
    """Determine the optimal retrieval strategy based on query signals.

    Args:
        query: User input query.

    Returns:
        A tuple of (strategy_name, reasoning, confidence):
            - strategy_name: one of "pageindex", "bm25", "vector", "hybrid"
            - reasoning: concise explanation of the decision
            - confidence: float score between 0.0 and 1.0
    """
    clean_q = query.strip()
    if not clean_q:
        return "hybrid", "Empty query default", 0.5

    lower_q = clean_q.lower()

    # 1. Structural cues -> PageIndex
    structural_patterns = [
        r"\b(?:page|pg|p\.?)\s*\d+\b",
        r"\b(?:section|chapter|part)\s*(?:\d+|[ivx]+|[a-z])\b",
        r"\b(?:table of contents|appendix|outline|heading|executive summary)\b",
    ]
    for pattern in structural_patterns:
        if re.search(pattern, lower_q):
            return (
                "pageindex",
                f"Query references structural document elements matching pattern '{pattern}'",
                0.90,
            )

    # 2. Exact identifiers, quotes, or code-like patterns -> BM25
    has_quotes = bool(re.search(r'["\'].+?["\']', clean_q))
    has_technical_id = bool(re.search(r"\b[A-Za-z0-9_-]+[_\d][A-Za-z0-9_-]*\b", clean_q))
    has_exact_keyword_prompt = any(
        lower_q.startswith(prefix)
        for prefix in ["find exact", "keyword:", "locate term", "contains:"]
    )
    if has_quotes or has_technical_id or has_exact_keyword_prompt:
        return (
            "bm25",
            "Query contains exact quoted text, technical identifiers, or exact keyword matching cues",
            0.85,
        )

    # 3. High-level conceptual / semantic queries -> Vector
    conceptual_starters = [
        "what is the difference between",
        "explain how",
        "why does",
        "what does it mean",
        "summarize the concept of",
        "in simple terms",
    ]
    if any(lower_q.startswith(s) for s in conceptual_starters):
        return (
            "vector",
            "Query is a broad semantic/conceptual question suitable for dense embedding space",
            0.80,
        )

    # 4. Multi-faceted / default queries -> Hybrid
    return (
        "hybrid",
        "Query benefits from fused semantic and keyword retrieval via Reciprocal Rank Fusion",
        0.75,
    )
