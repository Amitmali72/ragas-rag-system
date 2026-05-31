"""Semantic retriever using FAISS vector similarity search.

Wraps a LangChain FAISS index and returns ranked results with
normalized similarity scores (L2 distance → [0, 1] similarity).
"""

from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS

from config.settings import TOP_K_SEMANTIC


class SemanticRetriever:
    """Retrieve documents by dense-vector semantic similarity.

    Uses the FAISS index's built-in similarity_search_with_score method,
    then converts raw L2 distances into a 0-to-1 similarity score.
    """

    def __init__(self, faiss_index: FAISS) -> None:
        """Initialise with an existing FAISS vector store.

        Args:
            faiss_index: A LangChain FAISS vector store instance.
        """
        self.faiss_index: FAISS = faiss_index

    def retrieve(
        self, query: str, top_k: int = TOP_K_SEMANTIC
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k semantically similar chunks for a query.

        Args:
            query: The user's natural-language question.
            top_k: Number of results to return.

        Returns:
            A list of result dicts, each containing:
                - document: the LangChain Document object
                - score: similarity score in [0, 1]  (higher = more similar)
                - rank: 1-based rank
                - retriever: literal "semantic"
        """
        # FAISS returns (Document, L2_distance) tuples
        raw_results = self.faiss_index.similarity_search_with_score(
            query, k=top_k
        )

        results: List[Dict[str, Any]] = []
        for rank, (doc, distance) in enumerate(raw_results, start=1):
            # Convert L2 distance to a similarity score: 1/(1+d)
            similarity_score: float = 1.0 / (1.0 + float(distance))

            results.append({
                "document": doc,
                "score": similarity_score,
                "rank": rank,
                "retriever": "semantic",
            })

        return results
