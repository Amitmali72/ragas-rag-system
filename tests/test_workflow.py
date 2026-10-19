"""Tests for the Adaptive LangGraph workflow in RAGNEXUS."""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from workflow.router import route_query_strategy
from workflow.nodes import grade_context_node, rewrite_query_node, fallback_node
from workflow.graph import build_adaptive_rag_graph, run_adaptive_rag


class TestWorkflowRouter:
    """Tests for deterministic routing signals."""

    def test_routes_to_pageindex_on_structural_cues(self) -> None:
        strategy, reason, conf = route_query_strategy("What is discussed on page 4?")
        assert strategy == "pageindex"
        assert conf >= 0.8

        strategy2, _, _ = route_query_strategy("Summarize the Executive Summary in section 2")
        assert strategy2 == "pageindex"

    def test_routes_to_bm25_on_exact_identifiers_or_quotes(self) -> None:
        strategy, reason, _ = route_query_strategy('Find the definition of "strict_grounding_v1"')
        assert strategy == "bm25"

        strategy2, _, _ = route_query_strategy("What is error code ERR_404_NOT_FOUND?")
        assert strategy2 == "bm25"

    def test_routes_to_vector_on_broad_conceptual_questions(self) -> None:
        strategy, reason, _ = route_query_strategy("Explain how neural networks learn patterns")
        assert strategy == "vector"

    def test_routes_to_hybrid_on_standard_queries(self) -> None:
        strategy, reason, _ = route_query_strategy("What is machine learning in business applications?")
        assert strategy == "hybrid"


class TestWorkflowNodes:
    """Tests for individual node functions."""

    def test_grade_context_relevant(self) -> None:
        state = {
            "query": "machine learning patterns",
            "retrieved_chunks": [
                {"document": Document(page_content="Machine learning detects patterns in training data.")}
            ],
            "diagnostic_trace": [],
        }
        graded = grade_context_node(state)
        assert graded["context_grade"] == "relevant"

    def test_grade_context_inadequate(self) -> None:
        state = {
            "query": "quantum computing qubits",
            "retrieved_chunks": [
                {"document": Document(page_content="Baking a chocolate cake requires flour and sugar.")}
            ],
            "diagnostic_trace": [],
        }
        graded = grade_context_node(state)
        assert graded["context_grade"] == "inadequate"

    def test_rewrite_query_increments_retry(self) -> None:
        state = {
            "query": "Please tell me in detail about neural network weights",
            "retry_count": 0,
            "query_history": [],
            "diagnostic_trace": [],
        }
        rewritten = rewrite_query_node(state)
        assert rewritten["retry_count"] == 1
        assert "neural network weights" in rewritten["query"]
        assert len(rewritten["query_history"]) == 1

    def test_fallback_node_returns_unfound(self) -> None:
        state = {"retry_count": 2, "diagnostic_trace": []}
        res = fallback_node(state)
        assert "not found" in res["answer"].lower()
        assert res["is_grounded"] is True


class TestAdaptiveGraphExecution:
    """Tests for compiled StateGraph end-to-end execution."""

    @pytest.fixture
    def mock_retrievers(self):
        doc1 = Document(
            page_content="Deep learning architectures use multiple layers for hierarchical feature extraction.",
            metadata={"source": "ai_book.pdf", "page_number": 5, "chunk_id": "ai_p5_c0"},
        )
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {"document": doc1, "score": 0.92, "rank": 1, "retriever": "mock"}
        ]
        return mock_retriever

    def test_successful_adaptive_retrieval(self, mock_retrievers) -> None:
        """Graph should route, retrieve, grade as relevant, generate answer, and verify grounding."""
        final_state = run_adaptive_rag(
            query="Explain how deep learning architectures use layers",
            semantic_retriever=mock_retrievers,
            hybrid_retriever=mock_retrievers,
            max_retries=2,
        )

        assert final_state["strategy"] in ["vector", "hybrid"]
        assert final_state["context_grade"] == "relevant"
        assert len(final_state["retrieved_chunks"]) > 0
        assert len(final_state["citations"]) > 0
        assert final_state["is_grounded"] is True
        assert len(final_state["diagnostic_trace"]) >= 4

    def test_strict_retry_limit_to_fallback(self) -> None:
        """When retriever returns irrelevant content, graph must loop at most max_retries then fallback."""
        irrelevant_doc = Document(
            page_content="Unrelated text about gardening and planting trees.",
            metadata={"source": "garden.txt", "page_number": 1, "chunk_id": "g1"},
        )
        always_irrelevant = MagicMock()
        always_irrelevant.retrieve.return_value = [
            {"document": irrelevant_doc, "score": 0.1, "rank": 1, "retriever": "mock"}
        ]

        final_state = run_adaptive_rag(
            query="quantum supremacy algorithms",
            hybrid_retriever=always_irrelevant,
            semantic_retriever=always_irrelevant,
            max_retries=2,
        )

        # Must exit cleanly at fallback
        assert final_state["retry_count"] == 2
        assert "not found in uploaded documents" in final_state["answer"].lower()
        assert any("Fallback triggered" in t for t in final_state["diagnostic_trace"])
