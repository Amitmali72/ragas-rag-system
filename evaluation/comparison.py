"""
LLM vs Hybrid RAG Comparison Module.

Runs a side-by-side evaluation of:
  1. **Baseline LLM** – direct question → answer (no retrieval).
  2. **Hybrid RAG**   – question → retrieve chunks → grounded answer.

Both answer sets are scored with RAGAS, and the percentage improvement
of RAG over the baseline is computed for each metric.

Usage:
    >>> from evaluation.comparison import run_comparison
    >>> results = run_comparison(
    ...     questions=["What is ML?"],
    ...     ground_truths=["ML is machine learning."],
    ...     hybrid_retriever=retriever,
    ...     llm=llm,
    ...     generate_answer_fn=generate_answer,
    ...     generate_direct_fn=generate_direct_answer,
    ...     evaluate_fn=evaluate_rag,
    ... )
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===================================================================
# Percentage-improvement helper
# ===================================================================

def calculate_improvement(baseline_score: float, rag_score: float) -> float:
    """Calculate percentage improvement of RAG over a baseline.

    Parameters
    ----------
    baseline_score : float
        The metric value from the baseline (direct LLM) approach.
    rag_score : float
        The metric value from the Hybrid RAG approach.

    Returns
    -------
    float
        Percentage improvement.  Positive means RAG is better.
        Returns ``0.0`` if the baseline score is zero to avoid
        division-by-zero.

    Examples
    --------
    >>> calculate_improvement(0.60, 0.78)
    30.0
    >>> calculate_improvement(0.0, 0.85)
    0.0
    """
    if baseline_score == 0.0:
        return 0.0
    return round(((rag_score - baseline_score) / baseline_score) * 100.0, 2)


# ===================================================================
# Core comparison runner
# ===================================================================

def run_comparison(
    questions: Any,
    ground_truths: Optional[List[str]] = None,
    hybrid_retriever: Optional[Any] = None,
    llm: Optional[Any] = None,
    generate_answer_fn: Optional[Callable[..., Tuple[str, float]]] = None,
    generate_direct_fn: Optional[Callable[..., Tuple[str, float]]] = None,
    evaluate_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    top_k_semantic: int = 5,
    top_k_bm25: int = 5,
) -> Dict[str, Any]:
    """Run a full Baseline-LLM vs Hybrid-RAG comparison.

    Supports both:
      - Raw evaluation: questions, ground_truths, hybrid_retriever, llm, generate_fns, evaluate_fn
      - Pre-computed comparison items list: a single list of dicts as the first argument,
        containing: question, ground_truth, direct_answer, rag_answer, direct_time_ms, rag_time_ms, contexts

    Parameters
    ----------
    questions : Any
        Evaluation questions, or a list of dictionaries with pre-computed outputs.
    ground_truths : List[str], optional
        Expected reference answers (one per question).
    hybrid_retriever : object, optional
        An instance of ``retrieval.hybrid_retriever.HybridRetriever``
    llm : object, optional
        A LangChain-compatible LLM (e.g. from ``llm.groq_llm``).
    generate_answer_fn : Callable, optional
        The RAG answer generator.
    generate_direct_fn : Callable, optional
        The direct / baseline answer generator.
    evaluate_fn : Callable, optional
        The RAGAS evaluation function.
    top_k_semantic : int
        Number of semantic (FAISS) results to retrieve per question.
    top_k_bm25 : int
        Number of BM25 keyword results to retrieve per question.

    Returns
    -------
    Dict[str, Any]
        A comparison results dictionary.
    """

    # ---- Handle list-of-dicts input format ----------------------------------
    if questions and isinstance(questions, list) and isinstance(questions[0], dict):
        items = questions
        
        # If evaluate_fn is not provided, use default evaluate_rag
        if evaluate_fn is None:
            from evaluation.ragas_evaluator import evaluate_rag as default_eval_fn
            evaluate_fn = default_eval_fn

        questions_list = [it.get("question", "") for it in items]
        ground_truths_list = [it.get("ground_truth", "") for it in items]
        direct_answers = [it.get("direct_answer", "") for it in items]
        direct_contexts = [["No context used – direct LLM answer."] for _ in items]
        rag_answers = [it.get("rag_answer", "") for it in items]
        rag_contexts = [it.get("contexts", []) for it in items]
        
        # Evaluate baseline
        baseline_metrics = evaluate_fn(
            questions=questions_list,
            answers=direct_answers,
            contexts=direct_contexts,
            ground_truths=ground_truths_list,
        )
        
        # Evaluate hybrid rag
        rag_metrics = evaluate_fn(
            questions=questions_list,
            answers=rag_answers,
            contexts=rag_contexts,
            ground_truths=ground_truths_list,
        )
        
        # Calculate improvements
        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        improvements: Dict[str, float] = {}
        for m in metric_names:
            b_score = baseline_metrics.get("aggregate", {}).get(m, 0.0)
            r_score = rag_metrics.get("aggregate", {}).get(m, 0.0)
            improvements[m] = calculate_improvement(b_score, r_score)
            
        # Merge per-question RAGAS scores into detail records
        per_question_details = []
        for idx, item in enumerate(items):
            detail = {
                "question": item.get("question", ""),
                "ground_truth": item.get("ground_truth", ""),
                "baseline_answer": item.get("direct_answer", ""),
                "baseline_latency_ms": round(item.get("direct_time_ms", 0.0), 2),
                "rag_answer": item.get("rag_answer", ""),
                "rag_latency_ms": round(item.get("rag_time_ms", 0.0), 2),
                "num_chunks_retrieved": len(item.get("contexts", [])),
            }
            if idx < len(baseline_metrics.get("per_question", [])):
                bq = baseline_metrics["per_question"][idx]
                detail["baseline_faithfulness"] = bq.get("faithfulness", 0.0)
                detail["baseline_answer_relevancy"] = bq.get("answer_relevancy", 0.0)
            if idx < len(rag_metrics.get("per_question", [])):
                rq = rag_metrics["per_question"][idx]
                detail["rag_faithfulness"] = rq.get("faithfulness", 0.0)
                detail["rag_answer_relevancy"] = rq.get("answer_relevancy", 0.0)
            per_question_details.append(detail)
            
        baseline_latencies = [float(it.get("direct_time_ms", 0.0)) for it in items]
        rag_latencies = [float(it.get("rag_time_ms", 0.0)) for it in items]
        
        return {
            "baseline": {
                "answers": direct_answers,
                "latencies_ms": baseline_latencies,
                "metrics": baseline_metrics,
            },
            "hybrid_rag": {
                "answers": rag_answers,
                "contexts": rag_contexts,
                "latencies_ms": rag_latencies,
                "metrics": rag_metrics,
            },
            "comparison": {
                "improvements": improvements,
            },
            "per_question": per_question_details,
            "summary": {
                "total_questions": len(items),
                "avg_baseline_latency_ms": round(
                    sum(baseline_latencies) / len(baseline_latencies) if baseline_latencies else 0.0, 2
                ),
                "avg_rag_latency_ms": round(
                    sum(rag_latencies) / len(rag_latencies) if rag_latencies else 0.0, 2
                ),
                "direct_llm": baseline_metrics.get("aggregate", {}),
                "hybrid_rag": rag_metrics.get("aggregate", {}),
            },
        }

    # ---- Validate inputs for standard workflow -------------------------------
    if not questions:
        raise ValueError("'questions' must not be empty.")
    if ground_truths is None or len(questions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch or ground_truths missing: {len(questions)} questions vs "
            f"{len(ground_truths) if ground_truths else 0} ground truths."
        )
    if evaluate_fn is None:
        from evaluation.ragas_evaluator import evaluate_rag as default_eval_fn
        evaluate_fn = default_eval_fn

    # ---- Collect answers for both approaches --------------------------------
    baseline_answers: List[str] = []
    baseline_latencies: List[float] = []
    baseline_contexts: List[List[str]] = []  # empty contexts for baseline

    rag_answers: List[str] = []
    rag_latencies: List[float] = []
    rag_contexts: List[List[str]] = []

    per_question_details: List[Dict[str, Any]] = []

    for idx, question in enumerate(questions):
        logger.info("Processing question %d/%d: %s", idx + 1, len(questions), question[:60])

        # --- Baseline: direct LLM answer ---
        t0 = time.time()
        try:
            baseline_answer, baseline_ms = generate_direct_fn(question, llm)
        except Exception as exc:
            logger.error("Baseline generation failed for Q%d: %s", idx + 1, exc)
            baseline_answer = f"[Error] {exc}"
            baseline_ms = 0.0
        baseline_elapsed = (time.time() - t0) * 1000  # total wall-clock ms
        baseline_answers.append(baseline_answer)
        baseline_latencies.append(baseline_ms)
        # Baseline has no retrieval context
        baseline_contexts.append(["No context used – direct LLM answer."])

        # --- Hybrid RAG: retrieve then generate ---
        t0 = time.time()
        try:
            results, retrieval_stats = hybrid_retriever.retrieve(
                query=question,
                top_k_semantic=top_k_semantic,
                top_k_bm25=top_k_bm25,
            )
            # Extract chunk texts from retrieval results
            context_chunks = [
                res["document"].page_content
                for res in results
                if hasattr(res.get("document"), "page_content")
            ]
            if not context_chunks:
                context_chunks = ["No relevant context found."]

            rag_answer, rag_ms = generate_answer_fn(question, context_chunks, llm)
        except Exception as exc:
            logger.error("RAG generation failed for Q%d: %s", idx + 1, exc)
            rag_answer = f"[Error] {exc}"
            rag_ms = 0.0
            context_chunks = ["Retrieval failed."]
        rag_elapsed = (time.time() - t0) * 1000
        rag_answers.append(rag_answer)
        rag_latencies.append(rag_ms)
        rag_contexts.append(context_chunks)

        # --- Per-question detail record ---
        per_question_details.append({
            "question": question,
            "ground_truth": ground_truths[idx],
            "baseline_answer": baseline_answer,
            "baseline_latency_ms": round(baseline_ms, 2),
            "rag_answer": rag_answer,
            "rag_latency_ms": round(rag_ms, 2),
            "num_chunks_retrieved": len(context_chunks),
        })

    # ---- Evaluate both approaches with RAGAS --------------------------------
    logger.info("Running RAGAS evaluation on baseline answers...")
    baseline_metrics = evaluate_fn(
        questions=questions,
        answers=baseline_answers,
        contexts=baseline_contexts,
        ground_truths=ground_truths,
    )

    logger.info("Running RAGAS evaluation on Hybrid RAG answers...")
    rag_metrics = evaluate_fn(
        questions=questions,
        answers=rag_answers,
        contexts=rag_contexts,
        ground_truths=ground_truths,
    )

    # ---- Calculate improvements per metric ----------------------------------
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    improvements: Dict[str, float] = {}
    for metric_name in metric_names:
        b_score = baseline_metrics.get("aggregate", {}).get(metric_name, 0.0)
        r_score = rag_metrics.get("aggregate", {}).get(metric_name, 0.0)
        improvements[metric_name] = calculate_improvement(b_score, r_score)

    # ---- Merge per-question RAGAS scores into detail records ----------------
    for idx, detail in enumerate(per_question_details):
        if idx < len(baseline_metrics.get("per_question", [])):
            bq = baseline_metrics["per_question"][idx]
            detail["baseline_faithfulness"] = bq.get("faithfulness", 0.0)
            detail["baseline_answer_relevancy"] = bq.get("answer_relevancy", 0.0)
        if idx < len(rag_metrics.get("per_question", [])):
            rq = rag_metrics["per_question"][idx]
            detail["rag_faithfulness"] = rq.get("faithfulness", 0.0)
            detail["rag_answer_relevancy"] = rq.get("answer_relevancy", 0.0)

    # ---- Build final results dict -------------------------------------------
    return {
        "baseline": {
            "answers": baseline_answers,
            "latencies_ms": baseline_latencies,
            "metrics": baseline_metrics,
        },
        "hybrid_rag": {
            "answers": rag_answers,
            "contexts": rag_contexts,
            "latencies_ms": rag_latencies,
            "metrics": rag_metrics,
        },
        "comparison": {
            "improvements": improvements,
        },
        "per_question": per_question_details,
        "summary": {
            "total_questions": len(questions),
            "avg_baseline_latency_ms": round(
                sum(baseline_latencies) / len(baseline_latencies), 2
            ),
            "avg_rag_latency_ms": round(
                sum(rag_latencies) / len(rag_latencies), 2
            ),
            "direct_llm": baseline_metrics.get("aggregate", {}),
            "hybrid_rag": rag_metrics.get("aggregate", {}),
        },
    }


# ===================================================================
# DataFrame formatting
# ===================================================================

def format_comparison_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Create a side-by-side comparison DataFrame from comparison results.

    Parameters
    ----------
    results : Dict[str, Any]
        Output of :func:`run_comparison`.

    Returns
    -------
    pd.DataFrame
        A table with one row per metric showing baseline score,
        RAG score, and percentage improvement.
    """

    baseline_agg = results.get("baseline", {}).get("metrics", {}).get("aggregate", {})
    rag_agg = results.get("hybrid_rag", {}).get("metrics", {}).get("aggregate", {})
    improvements = results.get("comparison", {}).get("improvements", {})

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    pretty_names = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }

    rows: List[Dict[str, Any]] = []
    for metric in metric_names:
        rows.append({
            "Metric": pretty_names.get(metric, metric),
            "Baseline (LLM)": round(baseline_agg.get(metric, 0.0), 4),
            "Hybrid RAG": round(rag_agg.get(metric, 0.0), 4),
            "Improvement (%)": improvements.get(metric, 0.0),
        })

    return pd.DataFrame(rows)
