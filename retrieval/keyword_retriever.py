"""Keyword-based retriever using BM25 (Okapi BM25).

Provides sparse / lexical retrieval to complement the dense semantic
retriever, capturing exact keyword matches that embeddings might miss.
"""

from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config.settings import TOP_K_BM25


class KeywordRetriever:
    """Retrieve documents by BM25 keyword relevance scoring.

    The BM25 index is built once at construction time from the full
    corpus of LangChain Document objects.
    """

    def __init__(self, documents: List[Document]) -> None:
        """Build a BM25 index over the provided documents.

        Tokenisation is simple whitespace splitting + lower-casing,
        which works well for English text without extra dependencies.

        Args:
            documents: List of LangChain Document objects to index.
        """
        self.documents: List[Document] = documents

        # Tokenize every document's text
        self.tokenized_corpus: List[List[str]] = [
            doc.page_content.lower().split() for doc in documents
        ]

        # Build BM25 index
        self.bm25: BM25Okapi = BM25Okapi(self.tokenized_corpus)

    def retrieve(
        self, query: str, top_k: int = TOP_K_BM25
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k BM25-scored chunks for a query.

        Args:
            query: The user's natural-language question.
            top_k: Number of results to return.

        Returns:
            A list of result dicts, each containing:
                - document: the LangChain Document object
                - score: BM25 relevance score (unnormalized, higher = better)
                - rank: 1-based rank
                - retriever: literal "keyword"
        """
        # Tokenize the query the same way as the corpus
        query_tokens: List[str] = query.lower().split()

        # Get BM25 scores for every document
        scores: np.ndarray = self.bm25.get_scores(query_tokens)

        # Get indices of top-k scores (descending)
        top_indices: np.ndarray = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            score_value: float = float(scores[idx])

            # Skip documents with zero relevance
            if score_value <= 0:
                continue

            results.append({
                "document": self.documents[idx],
                "score": score_value,
                "rank": rank,
                "retriever": "keyword",
            })

        return results
