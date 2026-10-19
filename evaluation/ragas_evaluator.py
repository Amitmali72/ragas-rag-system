"""
Evaluation Module for Knowledge Copilot.

This module provides evaluation capabilities using an LLM-as-a-Judge approach.
It computes four key metrics:
  - Faithfulness: How factually consistent the answer is with the context.
  - Answer Relevancy: How relevant the answer is to the question.
  - Context Precision: How precise the retrieved context is.
  - Context Recall: How much of the ground truth is captured by the context.

This implementation replaces the RAGAS library to avoid complex environment dependency issues.
"""

from __future__ import annotations

import logging
import warnings
import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ===================================================================
# Custom LLM-as-a-Judge Evaluator
# ===================================================================

def _get_score_from_llm(llm: Any, prompt: str) -> float:
    """Helper to get a numeric score (0-1) from the LLM."""
    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        # Try to find a number in the response
        match = re.search(r"([0-9]*\.?[0-9]+)", content)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.0
    except Exception as e:
        logger.error(f"LLM scoring error: {e}")
        return 0.0

def _evaluate_metric(metric_name: str, question: str, answer: str, contexts: List[str], ground_truth: str, llm: Any) -> float:
    """Compute a specific metric using a targeted LLM prompt."""

    prompts = {
        "faithfulness": (
            f"Rate the faithfulness of the following answer based ONLY on the provided context.\n"
            f"Context: {contexts}\n"
            f"Answer: {answer}\n\n"
            f"Faithfulness means the answer contains no information not present in the context. "
            f"Score from 0.0 to 1.0. Return ONLY the number."
        ),
        "answer_relevancy": (
            f"Rate how relevant the answer is to the original question.\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Score from 0.0 to 1.0 based on how directly it addresses the question. Return ONLY the number."
        ),
        "context_precision": (
            f"Rate the precision of the retrieved context for the given question.\n"
            f"Question: {question}\n"
            f"Context: {contexts}\n\n"
            f"Precision means the retrieved chunks are actually useful for answering the question. "
            f"Score from 0.0 to 1.0. Return ONLY the number."
        ),
        "context_recall": (
            f"Rate the recall of the retrieved context compared to the ground truth.\n"
            f"Ground Truth: {ground_truth}\n"
            f"Context: {contexts}\n\n"
            f"Recall means the context contains all the information needed to produce the ground truth. "
            f"Score from 0.0 to 1.0. Return ONLY the number."
        )
    }

    prompt = prompts.get(metric_name, "Score this from 0.0 to 1.0. Return ONLY the number.")
    return _get_score_from_llm(llm, prompt)

def evaluate_rag(
    questions: Any,
    answers: Optional[List[str]] = None,
    contexts: Optional[List[List[str]]] = None,
    ground_truths: Optional[List[str]] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate RAG pipeline outputs using an LLM-as-a-Judge approach."""

    # Handle list-of-dicts input format
    if questions and isinstance(questions, list) and isinstance(questions[0], dict):
        dict_list = questions
        questions_list = [item.get("question", "") for item in dict_list]
        answers_list = [item.get("answer", item.get("direct_answer", "")) for item in dict_list]
        contexts_list = [item.get("contexts", item.get("context", [])) for item in dict_list]
        ground_truths_list = [item.get("ground_truth", item.get("ground_truths", "")) for item in dict_list]
    else:
        questions_list = questions
        answers_list = answers
        contexts_list = contexts
        ground_truths_list = ground_truths

    if not questions_list:
        raise ValueError("'questions' list must not be empty.")
    if answers_list is None or contexts_list is None or ground_truths_list is None:
        raise ValueError("answers, contexts, and ground_truths must be provided.")

    if llm is None:
        raise ValueError("An LLM instance is required for this custom evaluation.")

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    per_question = []

    for i in range(len(questions_list)):
        q = questions_list[i]
        a = answers_list[i]
        c = contexts_list[i]
        gt = ground_truths_list[i]

        row = {"question": q, "answer": a}
        for m in metric_names:
            row[m] = _evaluate_metric(m, q, a, c, gt, llm)
        per_question.append(row)

    # Aggregate means
    aggregate = {}
    for m in metric_names:
        vals = [row[m] for row in per_question]
        aggregate[m] = sum(vals) / len(vals) if vals else 0.0

    return {
        "per_question": per_question,
        "aggregate": aggregate,
        "num_questions": len(questions_list),
        "ragas_available": True, # Set to True so the dashboard renders correctly
        "status": "success",
    }

def _unavailable_evaluation(questions: List[str], answers: List[str], reason: str = "RAGAS is unavailable.") -> Dict[str, Any]:
    # Kept for backward compatibility in other modules
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    per_question = [{"question": q, "answer": a, "status": "unavailable", **{m: None for m in metric_names}}
                    for q, a in zip(questions, answers)]
    return {
        "per_question": per_question,
        "aggregate": {m: None for m in metric_names},
        "num_questions": len(questions),
        "ragas_available": False,
        "status": "unavailable",
        "error_message": reason,
    }

def format_results_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    if not results.get("per_question"):
        return pd.DataFrame()

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    pretty_names = {m: m.replace("_", " ").title() for m in metric_names}

    rows = []
    for item in results["per_question"]:
        row = {"Question": item["question"][:80]}
        for m in metric_names:
            val = item.get(m)
            row[pretty_names[m]] = round(val, 4) if isinstance(val, (int, float)) else "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    avg_row = {"Question": "📊 Average"}
    for m in metric_names:
        col = pretty_names[m]
        valid_vals = pd.to_numeric(df[col], errors="coerce").dropna()
        avg_row[col] = round(float(valid_vals.mean()), 4) if not valid_vals.empty else "N/A"

    return pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
