"""4-Way Retrieval Benchmark Suite for RAGNEXUS.

Executes a fair, standardized comparison of all four retrieval strategies:
  1. Vector Retrieval (Semantic dense search via FAISS)
  2. BM25 Retrieval (Keyword sparse search via Rank-BM25)
  3. PageIndex Retrieval (Structure-aware outline & page navigation)
  4. Hybrid Retrieval (Reciprocal Rank Fusion k=60)

Evaluates identical questions under standardized top_k and model settings,
measuring latency, chunk counts, source precision, and grounding without
fabricating scores.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
from langchain_core.documents import Document

from llm.groq_llm import generate_answer


class FourWayBenchmarkRunner:
    """Benchmark runner comparing Vector, BM25, PageIndex, and Hybrid retrieval."""

    STRATEGIES = ["vector", "bm25", "pageindex", "hybrid"]

    def __init__(
        self,
        semantic_retriever: Any = None,
        keyword_retriever: Any = None,
        pageindex_retriever: Any = None,
        hybrid_retriever: Any = None,
        llm: Any = None,
        top_k: int = 5,
    ) -> None:
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.pageindex_retriever = pageindex_retriever
        self.hybrid_retriever = hybrid_retriever
        self.llm = llm
        self.top_k = top_k

    def run_strategy_retrieval(
        self,
        strategy: str,
        query: str,
    ) -> Tuple[List[Dict[str, Any]], float, Optional[str]]:
        """Run single retrieval strategy, capturing results, latency, and errors."""
        start_t = time.time()
        error = None
        chunks: List[Dict[str, Any]] = []

        try:
            if strategy == "vector":
                if self.semantic_retriever:
                    chunks = self.semantic_retriever.retrieve(query, top_k=self.top_k)
                else:
                    error = "Semantic retriever not configured"

            elif strategy == "bm25":
                if self.keyword_retriever:
                    chunks = self.keyword_retriever.retrieve(query, top_k=self.top_k)
                else:
                    error = "Keyword retriever not configured"

            elif strategy == "pageindex":
                if self.pageindex_retriever:
                    chunks = self.pageindex_retriever.retrieve(query, top_k=self.top_k)
                else:
                    error = "PageIndex retriever not configured"

            elif strategy == "hybrid":
                if self.hybrid_retriever:
                    raw_res = self.hybrid_retriever.retrieve(query, top_k_final=self.top_k)
                    chunks = raw_res[0] if isinstance(raw_res, tuple) else raw_res
                else:
                    error = "Hybrid retriever not configured"

            else:
                error = f"Unknown strategy: '{strategy}'"

        except Exception as exc:
            error = str(exc)

        elapsed_ms = round((time.time() - start_t) * 1000.0, 2)
        return chunks, elapsed_ms, error

    def run_benchmark(
        self,
        test_cases: List[Dict[str, str]],
        generate_answers: bool = False,
    ) -> Dict[str, Any]:
        """Run full 4-way benchmark across all test cases.

        Args:
            test_cases: List of dicts, each containing 'question' and optionally 'ground_truth'.
            generate_answers: If True and LLM is present, generate answers for each strategy.

        Returns:
            Dictionary containing per_test_case records, per_strategy summary, and metadata.
        """
        if not test_cases:
            raise ValueError("Test cases list must not be empty.")

        benchmark_results: List[Dict[str, Any]] = []

        for case_idx, item in enumerate(test_cases, start=1):
            question = item.get("question", "")
            ground_truth = item.get("ground_truth", "")

            case_entry: Dict[str, Any] = {
                "case_id": case_idx,
                "question": question,
                "ground_truth": ground_truth,
                "strategies": {},
            }

            for strategy in self.STRATEGIES:
                chunks, ret_time, error = self.run_strategy_retrieval(strategy, question)

                answer = ""
                gen_time = 0.0

                if generate_answers and self.llm and not error and chunks:
                    try:
                        context_docs = [c["document"] for c in chunks if "document" in c]
                        ans, g_time = generate_answer(question, context_docs, self.llm)
                        answer = ans
                        gen_time = g_time
                    except Exception as exc:
                        answer = f"Generation error: {exc}"

                # Calculate lexical overlap with ground truth if available
                ground_truth_overlap = 0.0
                if ground_truth and chunks:
                    gt_words = set(ground_truth.lower().split())
                    chunk_text = " ".join(
                        c["document"].page_content.lower() for c in chunks if "document" in c
                    )
                    chunk_words = set(chunk_text.split())
                    overlap_count = len(gt_words & chunk_words)
                    ground_truth_overlap = round(overlap_count / len(gt_words), 4) if gt_words else 0.0

                sources = list({
                    c["document"].metadata.get("source", "Unknown")
                    for c in chunks
                    if "document" in c and hasattr(c["document"], "metadata")
                })

                case_entry["strategies"][strategy] = {
                    "num_chunks": len(chunks),
                    "retrieval_time_ms": ret_time,
                    "generation_time_ms": gen_time,
                    "total_time_ms": round(ret_time + gen_time, 2),
                    "sources": sources,
                    "ground_truth_overlap": ground_truth_overlap,
                    "answer": answer,
                    "error": error,
                    "chunks_preview": [
                        {
                            "text": c["document"].page_content[:120] + "..." if "document" in c else "",
                            "source": c["document"].metadata.get("source", "") if "document" in c else "",
                            "score": c.get("score", 0.0),
                        }
                        for c in chunks[:3]
                    ],
                }

            benchmark_results.append(case_entry)

        # Build summary statistics
        summary: Dict[str, Any] = {}
        for strategy in self.STRATEGIES:
            latencies = [
                r["strategies"][strategy]["retrieval_time_ms"]
                for r in benchmark_results
                if r["strategies"][strategy]["error"] is None
            ]
            chunk_counts = [
                r["strategies"][strategy]["num_chunks"]
                for r in benchmark_results
            ]
            overlaps = [
                r["strategies"][strategy]["ground_truth_overlap"]
                for r in benchmark_results
            ]
            error_count = sum(
                1 for r in benchmark_results if r["strategies"][strategy]["error"] is not None
            )

            summary[strategy] = {
                "avg_retrieval_time_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "avg_chunks_retrieved": round(sum(chunk_counts) / len(chunk_counts), 2) if chunk_counts else 0.0,
                "avg_ground_truth_overlap": round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0,
                "error_count": error_count,
                "successful_runs": len(benchmark_results) - error_count,
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "total_test_cases": len(test_cases),
            "top_k": self.top_k,
            "summary": summary,
            "results": benchmark_results,
        }

    @staticmethod
    def format_summary_dataframe(benchmark_output: Dict[str, Any]) -> pd.DataFrame:
        """Format benchmark summary into a readable comparison DataFrame."""
        summary = benchmark_output.get("summary", {})
        rows: List[Dict[str, Any]] = []

        pretty_names = {
            "vector": "1. Vector (Semantic FAISS)",
            "bm25": "2. BM25 (Keyword Lexical)",
            "pageindex": "3. PageIndex (Structure-Aware)",
            "hybrid": "4. Hybrid (Dense + Sparse RRF)",
        }

        for strat, stats in summary.items():
            rows.append({
                "Retrieval Strategy": pretty_names.get(strat, strat.upper()),
                "Avg Latency (ms)": stats.get("avg_retrieval_time_ms", 0.0),
                "Avg Chunks": stats.get("avg_chunks_retrieved", 0.0),
                "GT Overlap": stats.get("avg_ground_truth_overlap", 0.0),
                "Errors": stats.get("error_count", 0),
                "Success": f"{stats.get('successful_runs', 0)}/{benchmark_output.get('total_test_cases', 0)}",
            })

        return pd.DataFrame(rows)

    @staticmethod
    def save_benchmark_report(benchmark_output: Dict[str, Any], output_dir: str = "data/benchmarks") -> str:
        """Save benchmark run output to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_4way_{timestamp}.json"
        path = os.path.join(output_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(benchmark_output, f, indent=2)

        return path
