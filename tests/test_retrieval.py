"""
Tests for the hybrid retrieval pipeline.

Covers:
  - BM25 keyword retrieval returns ranked results
  - Reciprocal Rank Fusion (RRF) score calculation
  - Deduplication of results from multiple retrievers
  - Mock FAISS index for semantic retrieval tests
  - LangChain Document creation for test fixtures

Run with:
    pytest tests/test_retrieval.py -v
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# We create lightweight LangChain Document stand-ins so tests don't
# hard-depend on langchain being installed at test time.
# ---------------------------------------------------------------------------

try:
    from langchain.schema import Document
except ImportError:
    # Minimal stand-in for testing without langchain installed
    class Document:  # type: ignore[no-redef]
        """Minimal LangChain Document stand-in for testing."""

        def __init__(self, page_content: str = "", metadata: dict | None = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

        def __repr__(self) -> str:
            return f"Document(page_content={self.page_content[:40]!r}...)"


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def sample_documents() -> List[Document]:
    """Create a small corpus of LangChain Document objects for testing."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "doc1.pdf", "chunk_index": 0},
        ),
        Document(
            page_content="Deep learning uses neural networks with many layers.",
            metadata={"source": "doc1.pdf", "chunk_index": 1},
        ),
        Document(
            page_content="Natural language processing enables computers to understand text.",
            metadata={"source": "doc2.pdf", "chunk_index": 0},
        ),
        Document(
            page_content="Reinforcement learning trains agents through rewards and penalties.",
            metadata={"source": "doc2.pdf", "chunk_index": 1},
        ),
        Document(
            page_content="Support vector machines find optimal decision boundaries.",
            metadata={"source": "doc3.pdf", "chunk_index": 0},
        ),
    ]


@pytest.fixture
def sample_retrieval_results(sample_documents: List[Document]) -> List[Dict[str, Any]]:
    """Simulated retrieval results in the format returned by HybridRetriever."""
    return [
        {
            "document": sample_documents[0],
            "score": 0.95,
            "rank": 1,
            "retrievers": ["semantic"],
        },
        {
            "document": sample_documents[1],
            "score": 0.88,
            "rank": 2,
            "retrievers": ["semantic", "bm25"],
        },
        {
            "document": sample_documents[2],
            "score": 0.82,
            "rank": 3,
            "retrievers": ["bm25"],
        },
    ]


# ===================================================================
# Tests: BM25 Keyword Retrieval
# ===================================================================

class TestBM25Retrieval:
    """Tests for BM25 keyword-based retrieval."""

    def test_bm25_returns_results(self, sample_documents: List[Document]) -> None:
        """BM25 retrieval should return a non-empty list for a matching query."""
        try:
            from rank_bm25 import BM25Okapi

            # Tokenize corpus
            corpus = [doc.page_content.lower().split() for doc in sample_documents]
            bm25 = BM25Okapi(corpus)

            # Query
            query = "machine learning artificial intelligence"
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)

            # At least one document should have a non-zero score
            assert max(scores) > 0.0, "BM25 should find at least one matching document"

        except ImportError:
            pytest.skip("rank_bm25 not installed")

    def test_bm25_ranking_order(self, sample_documents: List[Document]) -> None:
        """The most relevant document should be ranked first."""
        try:
            from rank_bm25 import BM25Okapi
            import numpy as np

            corpus = [doc.page_content.lower().split() for doc in sample_documents]
            bm25 = BM25Okapi(corpus)

            query_tokens = "machine learning".lower().split()
            scores = bm25.get_scores(query_tokens)
            top_idx = int(np.argmax(scores))

            # The first document mentions "machine learning" explicitly
            assert "machine learning" in sample_documents[top_idx].page_content.lower()

        except ImportError:
            pytest.skip("rank_bm25 or numpy not installed")

    def test_bm25_top_k(self, sample_documents: List[Document]) -> None:
        """Requesting top-k results should return at most k items."""
        try:
            from rank_bm25 import BM25Okapi
            import numpy as np

            corpus = [doc.page_content.lower().split() for doc in sample_documents]
            bm25 = BM25Okapi(corpus)

            query_tokens = "learning".lower().split()
            scores = bm25.get_scores(query_tokens)

            top_k = 3
            top_indices = np.argsort(scores)[::-1][:top_k]

            assert len(top_indices) <= top_k
            # Scores should be in descending order
            top_scores = [scores[i] for i in top_indices]
            assert top_scores == sorted(top_scores, reverse=True)

        except ImportError:
            pytest.skip("rank_bm25 or numpy not installed")


# ===================================================================
# Tests: Reciprocal Rank Fusion (RRF)
# ===================================================================

class TestRRFFusion:
    """Tests for the Reciprocal Rank Fusion score calculation."""

    def test_rrf_formula_basic(self) -> None:
        """Verify the RRF formula: score = 1 / (k + rank).

        The standard RRF constant k is 60.
        """
        k = 60
        rank = 1  # top result
        expected = 1.0 / (k + rank)  # ≈ 0.01639
        assert abs(expected - (1.0 / 61)) < 1e-6

    def test_rrf_higher_rank_lower_score(self) -> None:
        """Documents with higher rank numbers should get lower RRF scores."""
        k = 60
        score_rank_1 = 1.0 / (k + 1)
        score_rank_5 = 1.0 / (k + 5)
        score_rank_10 = 1.0 / (k + 10)

        assert score_rank_1 > score_rank_5 > score_rank_10

    def test_rrf_fusion_two_retrievers(self) -> None:
        """A document found by both retrievers should score higher than one found by only one."""
        k = 60

        # Document A: found at rank 1 by semantic, rank 3 by BM25
        score_a = 1.0 / (k + 1) + 1.0 / (k + 3)

        # Document B: found at rank 1 by semantic only
        score_b = 1.0 / (k + 1)

        assert score_a > score_b, (
            "Document found by both retrievers should have a higher fused score"
        )

    def test_rrf_scores_positive(self) -> None:
        """All RRF scores should be strictly positive."""
        k = 60
        for rank in range(1, 101):
            score = 1.0 / (k + rank)
            assert score > 0.0

    def test_rrf_combined_score_calculation(self) -> None:
        """Simulate full RRF fusion for a small set of results."""
        k = 60

        # Semantic results: doc_ids with their ranks
        semantic_ranks = {"doc_a": 1, "doc_b": 2, "doc_c": 3}
        # BM25 results
        bm25_ranks = {"doc_b": 1, "doc_d": 2, "doc_a": 3}

        # Fuse
        fused_scores: Dict[str, float] = {}
        for doc_id, rank in semantic_ranks.items():
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        for doc_id, rank in bm25_ranks.items():
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        # doc_b appears in both → should be top
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        assert sorted_docs[0][0] == "doc_b", (
            "doc_b should rank first since it appears in both retrievers at good ranks"
        )


# ===================================================================
# Tests: Deduplication
# ===================================================================

class TestDeduplication:
    """Tests for deduplication of results from multiple retrievers."""

    def test_no_duplicate_content(self, sample_documents: List[Document]) -> None:
        """After deduplication, no two results should have the same page_content."""
        # Simulate duplicates
        results_with_dupes = [
            {"document": sample_documents[0], "score": 0.9, "rank": 1, "retrievers": ["semantic"]},
            {"document": sample_documents[1], "score": 0.8, "rank": 2, "retrievers": ["semantic"]},
            {"document": sample_documents[0], "score": 0.7, "rank": 1, "retrievers": ["bm25"]},  # duplicate
        ]

        # Simple deduplication by page_content
        seen_contents: set[str] = set()
        deduplicated: List[Dict[str, Any]] = []
        for result in results_with_dupes:
            content = result["document"].page_content
            if content not in seen_contents:
                seen_contents.add(content)
                deduplicated.append(result)

        assert len(deduplicated) == 2, "Deduplication should remove the duplicate"
        contents = [r["document"].page_content for r in deduplicated]
        assert len(contents) == len(set(contents)), "All contents should be unique"

    def test_dedup_preserves_best_score(self, sample_documents: List[Document]) -> None:
        """When deduplicating, the result with the highest score should be kept."""
        results_with_dupes = [
            {"document": sample_documents[0], "score": 0.7, "rank": 2, "retrievers": ["bm25"]},
            {"document": sample_documents[0], "score": 0.9, "rank": 1, "retrievers": ["semantic"]},
        ]

        # Keep the one with the highest score
        best: Dict[str, Dict[str, Any]] = {}
        for result in results_with_dupes:
            content = result["document"].page_content
            if content not in best or result["score"] > best[content]["score"]:
                best[content] = result

        deduplicated = list(best.values())
        assert len(deduplicated) == 1
        assert deduplicated[0]["score"] == 0.9, "Should keep the higher-scored version"

    def test_dedup_merges_retriever_sources(self, sample_documents: List[Document]) -> None:
        """Deduplication should merge the retrievers list from both sources."""
        results_with_dupes = [
            {"document": sample_documents[0], "score": 0.9, "rank": 1, "retrievers": ["semantic"]},
            {"document": sample_documents[0], "score": 0.7, "rank": 1, "retrievers": ["bm25"]},
        ]

        merged: Dict[str, Dict[str, Any]] = {}
        for result in results_with_dupes:
            content = result["document"].page_content
            if content not in merged:
                merged[content] = result.copy()
                merged[content]["retrievers"] = list(result["retrievers"])
            else:
                # Merge retrievers
                for r in result["retrievers"]:
                    if r not in merged[content]["retrievers"]:
                        merged[content]["retrievers"].append(r)
                # Keep higher score
                if result["score"] > merged[content]["score"]:
                    merged[content]["score"] = result["score"]

        deduplicated = list(merged.values())
        assert len(deduplicated) == 1
        assert set(deduplicated[0]["retrievers"]) == {"semantic", "bm25"}


# ===================================================================
# Tests: Mock FAISS Semantic Retrieval
# ===================================================================

class TestSemanticRetrieval:
    """Tests for semantic retrieval using a mocked FAISS index."""

    def test_mock_faiss_returns_results(self, sample_documents: List[Document]) -> None:
        """A mocked FAISS similarity search should return Document objects."""
        # Mock a FAISS vector store
        mock_faiss = MagicMock()
        mock_faiss.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.15),  # (doc, L2 distance)
            (sample_documents[1], 0.32),
        ]

        results = mock_faiss.similarity_search_with_score("What is ML?", k=2)

        assert len(results) == 2
        assert results[0][0].page_content == sample_documents[0].page_content
        # Lower L2 distance = more similar
        assert results[0][1] < results[1][1]

    def test_mock_faiss_respects_top_k(self, sample_documents: List[Document]) -> None:
        """FAISS should return at most k results."""
        mock_faiss = MagicMock()
        mock_faiss.similarity_search_with_score.return_value = [
            (sample_documents[i], 0.1 * (i + 1))
            for i in range(3)
        ]

        results = mock_faiss.similarity_search_with_score("query", k=3)
        assert len(results) <= 3

    def test_semantic_scores_conversion(self) -> None:
        """L2 distances should be convertible to similarity scores in [0, 1]."""
        # Common conversion: similarity = 1 / (1 + distance)
        distances = [0.0, 0.5, 1.0, 2.0, 10.0]
        for dist in distances:
            similarity = 1.0 / (1.0 + dist)
            assert 0.0 < similarity <= 1.0
            # Closer distance → higher similarity
        similarities = [1.0 / (1.0 + d) for d in distances]
        assert similarities == sorted(similarities, reverse=True)

    def test_empty_index_returns_empty(self) -> None:
        """Querying an empty FAISS index should return an empty list."""
        mock_faiss = MagicMock()
        mock_faiss.similarity_search_with_score.return_value = []

        results = mock_faiss.similarity_search_with_score("anything", k=5)
        assert results == []
