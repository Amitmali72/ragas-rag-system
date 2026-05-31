"""
Tests for the evaluation module.

Covers:
  - RAGAS evaluator (with simulated fallback)
  - Comparison runner (with mocked LLM and retriever)
  - Improvement calculation
  - DataFrame formatting

Run with:
    pytest tests/test_evaluation.py -v
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from evaluation.comparison import (
    calculate_improvement,
    format_comparison_dataframe,
    run_comparison,
)
from evaluation.ragas_evaluator import evaluate_rag, format_results_dataframe


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def sample_questions() -> List[str]:
    """A small set of test questions."""
    return [
        "What is machine learning?",
        "How does deep learning work?",
    ]


@pytest.fixture
def sample_answers() -> List[str]:
    """Answers corresponding to sample_questions."""
    return [
        "Machine learning is a type of AI that learns from data.",
        "Deep learning uses neural networks with multiple layers.",
    ]


@pytest.fixture
def sample_contexts() -> List[List[str]]:
    """Retrieved context chunks for each question."""
    return [
        [
            "Machine learning is a subset of artificial intelligence.",
            "ML algorithms learn patterns from training data.",
        ],
        [
            "Deep learning is a branch of machine learning.",
            "Neural networks have input, hidden, and output layers.",
        ],
    ]


@pytest.fixture
def sample_ground_truths() -> List[str]:
    """Ground-truth reference answers."""
    return [
        "Machine learning is AI that learns from data without explicit programming.",
        "Deep learning uses multi-layer neural networks to learn representations.",
    ]


@pytest.fixture
def mock_llm() -> MagicMock:
    """A mock LangChain LLM."""
    llm = MagicMock()
    llm.invoke.return_value = "Mocked LLM response."
    return llm


@pytest.fixture
def mock_retriever(sample_contexts: List[List[str]]) -> MagicMock:
    """A mock HybridRetriever that returns pre-defined chunks."""
    retriever = MagicMock()

    # Create mock Document objects
    mock_docs = []
    for ctx_list in sample_contexts:
        docs = []
        for chunk_text in ctx_list:
            doc = MagicMock()
            doc.page_content = chunk_text
            doc.metadata = {"source": "test.pdf"}
            docs.append(doc)
        mock_docs.append(docs)

    # Track call count to return different results per question
    call_count = {"n": 0}

    def mock_retrieve(query: str, top_k_semantic: int = 5, top_k_bm25: int = 5):
        idx = min(call_count["n"], len(mock_docs) - 1)
        call_count["n"] += 1
        results = [
            {
                "document": doc,
                "score": 0.9 - i * 0.1,
                "rank": i + 1,
                "retrievers": ["semantic", "bm25"],
            }
            for i, doc in enumerate(mock_docs[idx])
        ]
        stats = {"total_results": len(results), "retrieval_time_ms": 50.0}
        return results, stats

    retriever.retrieve = mock_retrieve
    return retriever


# ===================================================================
# Tests: RAGAS Evaluator
# ===================================================================

class TestRagasEvaluator:
    """Tests for evaluation.ragas_evaluator.evaluate_rag."""

    def test_evaluate_returns_dict(
        self,
        sample_questions: List[str],
        sample_answers: List[str],
        sample_contexts: List[List[str]],
        sample_ground_truths: List[str],
    ) -> None:
        """evaluate_rag should return a dict with expected keys."""
        results = evaluate_rag(
            questions=sample_questions,
            answers=sample_answers,
            contexts=sample_contexts,
            ground_truths=sample_ground_truths,
        )

        assert isinstance(results, dict)
        assert "per_question" in results
        assert "aggregate" in results
        assert "num_questions" in results
        assert results["num_questions"] == len(sample_questions)

    def test_aggregate_scores_in_range(
        self,
        sample_questions: List[str],
        sample_answers: List[str],
        sample_contexts: List[List[str]],
        sample_ground_truths: List[str],
    ) -> None:
        """All aggregate scores should be between 0 and 1."""
        results = evaluate_rag(
            questions=sample_questions,
            answers=sample_answers,
            contexts=sample_contexts,
            ground_truths=sample_ground_truths,
        )

        for metric, score in results["aggregate"].items():
            assert 0.0 <= score <= 1.0, f"{metric} score {score} out of range [0, 1]"

    def test_per_question_count_matches(
        self,
        sample_questions: List[str],
        sample_answers: List[str],
        sample_contexts: List[List[str]],
        sample_ground_truths: List[str],
    ) -> None:
        """per_question should have one entry per input question."""
        results = evaluate_rag(
            questions=sample_questions,
            answers=sample_answers,
            contexts=sample_contexts,
            ground_truths=sample_ground_truths,
        )

        assert len(results["per_question"]) == len(sample_questions)

    def test_single_question(self) -> None:
        """Should handle a single question without errors."""
        results = evaluate_rag(
            questions=["What is AI?"],
            answers=["AI is artificial intelligence."],
            contexts=[["Artificial intelligence is a field of computer science."]],
            ground_truths=["AI is artificial intelligence."],
        )
        assert results["num_questions"] == 1

    def test_empty_input_raises(self) -> None:
        """Empty input lists should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            evaluate_rag(
                questions=[],
                answers=[],
                contexts=[],
                ground_truths=[],
            )

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched input lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            evaluate_rag(
                questions=["Q1", "Q2"],
                answers=["A1"],  # wrong length
                contexts=[["C1"]],
                ground_truths=["GT1"],
            )

    def test_simulated_warning_present(
        self,
        sample_questions: List[str],
        sample_answers: List[str],
        sample_contexts: List[List[str]],
        sample_ground_truths: List[str],
    ) -> None:
        """When RAGAS is unavailable, results should contain a warning."""
        # Force RAGAS unavailable
        with patch("evaluation.ragas_evaluator.RAGAS_AVAILABLE", False):
            results = evaluate_rag(
                questions=sample_questions,
                answers=sample_answers,
                contexts=sample_contexts,
                ground_truths=sample_ground_truths,
            )
            # Should either have ragas_available=False or a warning key
            assert results.get("ragas_available") is False or "warning" in results


# ===================================================================
# Tests: DataFrame Formatting
# ===================================================================

class TestFormatResultsDataFrame:
    """Tests for format_results_dataframe."""

    def test_returns_dataframe(
        self,
        sample_questions: List[str],
        sample_answers: List[str],
        sample_contexts: List[List[str]],
        sample_ground_truths: List[str],
    ) -> None:
        """Should return a pandas DataFrame."""
        results = evaluate_rag(
            questions=sample_questions,
            answers=sample_answers,
            contexts=sample_contexts,
            ground_truths=sample_ground_truths,
        )
        df = format_results_dataframe(results)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_has_average_row(
        self,
        sample_questions: List[str],
        sample_answers: List[str],
        sample_contexts: List[List[str]],
        sample_ground_truths: List[str],
    ) -> None:
        """DataFrame should have an 'Average' row at the bottom."""
        results = evaluate_rag(
            questions=sample_questions,
            answers=sample_answers,
            contexts=sample_contexts,
            ground_truths=sample_ground_truths,
        )
        df = format_results_dataframe(results)

        last_row_question = df.iloc[-1]["Question"]
        assert "Average" in str(last_row_question)

    def test_empty_results(self) -> None:
        """format_results_dataframe with empty results should return empty DataFrame."""
        df = format_results_dataframe({"per_question": []})
        assert df.empty


# ===================================================================
# Tests: Improvement Calculation
# ===================================================================

class TestCalculateImprovement:
    """Tests for evaluation.comparison.calculate_improvement."""

    def test_positive_improvement(self) -> None:
        """RAG score > baseline should give positive improvement."""
        result = calculate_improvement(0.60, 0.78)
        assert result == 30.0

    def test_negative_improvement(self) -> None:
        """RAG score < baseline should give negative improvement."""
        result = calculate_improvement(0.80, 0.60)
        assert result < 0.0

    def test_no_improvement(self) -> None:
        """Equal scores should give 0% improvement."""
        result = calculate_improvement(0.75, 0.75)
        assert result == 0.0

    def test_zero_baseline(self) -> None:
        """Zero baseline should return 0.0 (avoid division by zero)."""
        result = calculate_improvement(0.0, 0.85)
        assert result == 0.0

    def test_perfect_improvement(self) -> None:
        """Going from 0.5 to 1.0 should be 100% improvement."""
        result = calculate_improvement(0.50, 1.0)
        assert result == 100.0


# ===================================================================
# Tests: Comparison Runner
# ===================================================================

class TestRunComparison:
    """Tests for evaluation.comparison.run_comparison."""

    def _make_generate_answer(self):
        """Create a mock generate_answer function."""
        def generate_answer(query: str, chunks: List[str], llm: Any) -> Tuple[str, float]:
            return f"RAG answer for: {query[:30]}", 150.0
        return generate_answer

    def _make_generate_direct(self):
        """Create a mock generate_direct_answer function."""
        def generate_direct(query: str, llm: Any) -> Tuple[str, float]:
            return f"Direct answer for: {query[:30]}", 100.0
        return generate_direct

    def _make_evaluate_fn(self):
        """Create a mock evaluate function that returns fixed scores."""
        def evaluate_fn(
            questions: List[str],
            answers: List[str],
            contexts: List[List[str]],
            ground_truths: List[str],
            **kwargs,
        ) -> Dict[str, Any]:
            n = len(questions)
            per_q = []
            for i in range(n):
                per_q.append({
                    "question": questions[i],
                    "answer": answers[i],
                    "faithfulness": 0.75,
                    "answer_relevancy": 0.80,
                    "context_precision": 0.70,
                    "context_recall": 0.65,
                })
            return {
                "per_question": per_q,
                "aggregate": {
                    "faithfulness": 0.75,
                    "answer_relevancy": 0.80,
                    "context_precision": 0.70,
                    "context_recall": 0.65,
                },
                "num_questions": n,
                "ragas_available": False,
            }
        return evaluate_fn

    def test_comparison_returns_expected_keys(
        self,
        sample_questions: List[str],
        sample_ground_truths: List[str],
        mock_retriever: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        """run_comparison should return a dict with baseline, hybrid_rag, comparison keys."""
        results = run_comparison(
            questions=sample_questions,
            ground_truths=sample_ground_truths,
            hybrid_retriever=mock_retriever,
            llm=mock_llm,
            generate_answer_fn=self._make_generate_answer(),
            generate_direct_fn=self._make_generate_direct(),
            evaluate_fn=self._make_evaluate_fn(),
        )

        assert "baseline" in results
        assert "hybrid_rag" in results
        assert "comparison" in results
        assert "per_question" in results

    def test_comparison_answer_counts(
        self,
        sample_questions: List[str],
        sample_ground_truths: List[str],
        mock_retriever: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        """Both baseline and RAG should produce one answer per question."""
        results = run_comparison(
            questions=sample_questions,
            ground_truths=sample_ground_truths,
            hybrid_retriever=mock_retriever,
            llm=mock_llm,
            generate_answer_fn=self._make_generate_answer(),
            generate_direct_fn=self._make_generate_direct(),
            evaluate_fn=self._make_evaluate_fn(),
        )

        assert len(results["baseline"]["answers"]) == len(sample_questions)
        assert len(results["hybrid_rag"]["answers"]) == len(sample_questions)

    def test_comparison_improvements_present(
        self,
        sample_questions: List[str],
        sample_ground_truths: List[str],
        mock_retriever: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        """Improvements dict should have all four RAGAS metric keys."""
        results = run_comparison(
            questions=sample_questions,
            ground_truths=sample_ground_truths,
            hybrid_retriever=mock_retriever,
            llm=mock_llm,
            generate_answer_fn=self._make_generate_answer(),
            generate_direct_fn=self._make_generate_direct(),
            evaluate_fn=self._make_evaluate_fn(),
        )

        improvements = results["comparison"]["improvements"]
        expected_metrics = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
        assert set(improvements.keys()) == expected_metrics

    def test_empty_questions_raises(
        self,
        mock_retriever: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        """Empty questions should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            run_comparison(
                questions=[],
                ground_truths=[],
                hybrid_retriever=mock_retriever,
                llm=mock_llm,
                generate_answer_fn=self._make_generate_answer(),
                generate_direct_fn=self._make_generate_direct(),
                evaluate_fn=self._make_evaluate_fn(),
            )


# ===================================================================
# Tests: Comparison DataFrame Formatting
# ===================================================================

class TestFormatComparisonDataFrame:
    """Tests for format_comparison_dataframe."""

    def test_returns_dataframe(self) -> None:
        """Should return a pandas DataFrame with expected columns."""
        mock_results = {
            "baseline": {
                "metrics": {
                    "aggregate": {
                        "faithfulness": 0.60,
                        "answer_relevancy": 0.65,
                        "context_precision": 0.55,
                        "context_recall": 0.50,
                    }
                }
            },
            "hybrid_rag": {
                "metrics": {
                    "aggregate": {
                        "faithfulness": 0.82,
                        "answer_relevancy": 0.88,
                        "context_precision": 0.78,
                        "context_recall": 0.72,
                    }
                }
            },
            "comparison": {
                "improvements": {
                    "faithfulness": 36.67,
                    "answer_relevancy": 35.38,
                    "context_precision": 41.82,
                    "context_recall": 44.0,
                }
            },
        }

        df = format_comparison_dataframe(mock_results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 4 metrics
        assert "Metric" in df.columns
        assert "Baseline (LLM)" in df.columns
        assert "Hybrid RAG" in df.columns
        assert "Improvement (%)" in df.columns

    def test_all_metrics_present(self) -> None:
        """All four RAGAS metrics should appear as rows."""
        mock_results = {
            "baseline": {"metrics": {"aggregate": {
                "faithfulness": 0.5, "answer_relevancy": 0.5,
                "context_precision": 0.5, "context_recall": 0.5,
            }}},
            "hybrid_rag": {"metrics": {"aggregate": {
                "faithfulness": 0.8, "answer_relevancy": 0.8,
                "context_precision": 0.8, "context_recall": 0.8,
            }}},
            "comparison": {"improvements": {
                "faithfulness": 60.0, "answer_relevancy": 60.0,
                "context_precision": 60.0, "context_recall": 60.0,
            }},
        }

        df = format_comparison_dataframe(mock_results)
        metrics_in_df = set(df["Metric"].tolist())
        expected = {"Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"}
        assert metrics_in_df == expected
