"""Tests for the 4-Way Retrieval Benchmark Suite in RAGNEXUS."""

import os
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from evaluation.benchmark_suite import FourWayBenchmarkRunner


@pytest.fixture
def sample_test_cases():
    return [
        {
            "question": "What is machine learning?",
            "ground_truth": "Machine learning is a subset of artificial intelligence.",
        },
        {
            "question": "What is discussed on page 2?",
            "ground_truth": "Page 2 discusses system architecture and vector stores.",
        },
    ]


@pytest.fixture
def mock_retrievers():
    doc1 = Document(
        page_content="Machine learning enables computers to learn from data.",
        metadata={"source": "ml_guide.pdf", "page_number": 1, "chunk_id": "c1"},
    )
    doc2 = Document(
        page_content="Section 2 covers system architecture and vector stores.",
        metadata={"source": "ml_guide.pdf", "page_number": 2, "chunk_id": "c2"},
    )

    vector_ret = MagicMock()
    vector_ret.retrieve.return_value = [{"document": doc1, "score": 0.88, "rank": 1, "retriever": "semantic"}]

    bm25_ret = MagicMock()
    bm25_ret.retrieve.return_value = [{"document": doc1, "score": 4.5, "rank": 1, "retriever": "keyword"}]

    pageindex_ret = MagicMock()
    pageindex_ret.retrieve.return_value = [{"document": doc2, "score": 0.95, "rank": 1, "retriever": "pageindex"}]

    hybrid_ret = MagicMock()
    hybrid_ret.retrieve.return_value = (
        [{"document": doc1, "score": 0.032, "rank": 1, "retrievers": ["semantic", "keyword"]}],
        {"retrieval_time_ms": 12.5},
    )

    return {
        "vector": vector_ret,
        "bm25": bm25_ret,
        "pageindex": pageindex_ret,
        "hybrid": hybrid_ret,
    }


class TestFourWayBenchmarkRunner:
    """Tests for 4-way comparative benchmark execution."""

    def test_evaluates_all_four_strategies(self, sample_test_cases, mock_retrievers) -> None:
        runner = FourWayBenchmarkRunner(
            semantic_retriever=mock_retrievers["vector"],
            keyword_retriever=mock_retrievers["bm25"],
            pageindex_retriever=mock_retrievers["pageindex"],
            hybrid_retriever=mock_retrievers["hybrid"],
            top_k=3,
        )

        output = runner.run_benchmark(sample_test_cases, generate_answers=False)

        assert output["total_test_cases"] == 2
        assert len(output["results"]) == 2

        # Check that every question was executed against all 4 strategies
        for item in output["results"]:
            strategies = item["strategies"]
            assert "vector" in strategies
            assert "bm25" in strategies
            assert "pageindex" in strategies
            assert "hybrid" in strategies

            for strat_name, strat_data in strategies.items():
                assert strat_data["num_chunks"] > 0
                assert strat_data["retrieval_time_ms"] >= 0.0
                assert strat_data["error"] is None
                assert isinstance(strat_data["sources"], list)

    def test_summary_and_dataframe_generation(self, sample_test_cases, mock_retrievers) -> None:
        runner = FourWayBenchmarkRunner(
            semantic_retriever=mock_retrievers["vector"],
            keyword_retriever=mock_retrievers["bm25"],
            pageindex_retriever=mock_retrievers["pageindex"],
            hybrid_retriever=mock_retrievers["hybrid"],
        )

        output = runner.run_benchmark(sample_test_cases)
        summary = output["summary"]

        for strat in ["vector", "bm25", "pageindex", "hybrid"]:
            assert strat in summary
            assert summary[strat]["successful_runs"] == 2
            assert summary[strat]["error_count"] == 0

        df = FourWayBenchmarkRunner.format_summary_dataframe(output)
        assert len(df) == 4
        assert "Retrieval Strategy" in df.columns
        assert "Avg Latency (ms)" in df.columns

    def test_transparent_error_reporting_for_missing_retriever(self, sample_test_cases) -> None:
        """When a retriever is missing, it should report the error rather than inventing fake results."""
        runner = FourWayBenchmarkRunner(
            semantic_retriever=None,  # Missing
            keyword_retriever=None,
            pageindex_retriever=None,
            hybrid_retriever=None,
        )

        output = runner.run_benchmark(sample_test_cases)
        vector_res = output["results"][0]["strategies"]["vector"]
        assert vector_res["error"] is not None
        assert "not configured" in vector_res["error"]
        assert output["summary"]["vector"]["error_count"] == 2

    def test_benchmark_persistence(self, sample_test_cases, mock_retrievers, tmp_path) -> None:
        runner = FourWayBenchmarkRunner(
            semantic_retriever=mock_retrievers["vector"],
            keyword_retriever=mock_retrievers["bm25"],
            pageindex_retriever=mock_retrievers["pageindex"],
            hybrid_retriever=mock_retrievers["hybrid"],
        )
        output = runner.run_benchmark(sample_test_cases)
        saved_path = FourWayBenchmarkRunner.save_benchmark_report(output, output_dir=str(tmp_path))

        assert os.path.exists(saved_path)
        assert os.path.getsize(saved_path) > 100

    def test_default_benchmark_cases_structure(self) -> None:
        from frontend.evaluation_dashboard import DEFAULT_BENCHMARK_CASES

        assert len(DEFAULT_BENCHMARK_CASES) >= 4
        for case in DEFAULT_BENCHMARK_CASES:
            assert "question" in case and len(case["question"]) > 10
            assert "ground_truth" in case and len(case["ground_truth"]) > 10
