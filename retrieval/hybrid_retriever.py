"""Hybrid retriever combining semantic and keyword search via Reciprocal Rank Fusion.

RRF is a simple, effective technique that merges ranked lists from different
retrievers without requiring score normalization.  The fused score for a
document is:  sum( 1 / (k + rank_i) )  across all retrievers that returned it.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

from retrieval.semantic_retriever import SemanticRetriever
from retrieval.keyword_retriever import KeywordRetriever
from config.settings import RRF_K, TOP_K_SEMANTIC, TOP_K_BM25


class HybridRetriever:
    """Fuse semantic (dense) and keyword (sparse) retrieval with RRF."""

    def __init__(
        self,
        semantic_retriever: Optional[SemanticRetriever] = None,
        keyword_retriever: Optional[KeywordRetriever] = None,
        faiss_index: Optional[Any] = None,
    ) -> None:
        """Store both sub-retrievers, supporting FAISS index input directly.

        Args:
            semantic_retriever: FAISS-based semantic retriever instance.
            keyword_retriever: BM25-based keyword retriever instance.
            faiss_index: A FAISS vector store instance. If provided, semantic_retriever is built from it.
        """
        if semantic_retriever is not None:
            self.semantic_retriever = semantic_retriever
        elif faiss_index is not None:
            from retrieval.semantic_retriever import SemanticRetriever as DefaultSemanticRetriever
            self.semantic_retriever = DefaultSemanticRetriever(faiss_index)
        else:
            raise ValueError("Either semantic_retriever or faiss_index must be supplied to HybridRetriever.")

        self.keyword_retriever = keyword_retriever

    # ----- internal helpers ------------------------------------------------

    @staticmethod
    def _content_hash(text: str) -> str:
        """Return an MD5 hex digest for a chunk's page_content."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        k: int = RRF_K,
    ) -> List[Dict[str, Any]]:
        """Merge two ranked result lists using Reciprocal Rank Fusion.

        For every unique document (identified by content hash), the RRF
        score is the sum of  1 / (k + rank)  across all retrievers that
        returned it.

        Args:
            semantic_results: Ranked results from the semantic retriever.
            keyword_results: Ranked results from the keyword retriever.
            k: RRF constant (default 60).  Higher values dampen rank differences.

        Returns:
            A single fused list of result dicts, sorted by descending RRF score.
            Each dict contains: document, score (RRF), rank, retrievers (list).
        """
        # Accumulate scores keyed by content hash
        fused_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}      # hash → document dict
        retriever_map: Dict[str, List[str]] = {}      # hash → list of retriever names

        for result in semantic_results:
            content_hash = self._content_hash(result["document"].page_content)
            rrf_score = 1.0 / (k + result["rank"])
            fused_scores[content_hash] = fused_scores.get(content_hash, 0.0) + rrf_score
            doc_map[content_hash] = result["document"]
            retriever_map.setdefault(content_hash, []).append("semantic")

        for result in keyword_results:
            content_hash = self._content_hash(result["document"].page_content)
            rrf_score = 1.0 / (k + result["rank"])
            fused_scores[content_hash] = fused_scores.get(content_hash, 0.0) + rrf_score
            doc_map.setdefault(content_hash, result["document"])
            retriever_map.setdefault(content_hash, []).append("keyword")

        # Sort by fused RRF score (descending)
        sorted_hashes = sorted(
            fused_scores.keys(), key=lambda h: fused_scores[h], reverse=True
        )

        fused_results: List[Dict[str, Any]] = []
        for rank, content_hash in enumerate(sorted_hashes, start=1):
            fused_results.append({
                "document": doc_map[content_hash],
                "score": fused_scores[content_hash],
                "rank": rank,
                "retrievers": list(set(retriever_map[content_hash])),
            })

        return fused_results

    def _deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate chunks based on content hash.

        Only the first (highest-ranked) occurrence is kept.

        Args:
            results: List of result dicts.

        Returns:
            De-duplicated list preserving original order.
        """
        seen: set[str] = set()
        unique: List[Dict[str, Any]] = []

        for result in results:
            content_hash = self._content_hash(result["document"].page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(result)

        return unique

    # ----- public API ------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k_semantic: int = TOP_K_SEMANTIC,
        top_k_bm25: int = TOP_K_BM25,
        top_k_final: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run full hybrid retrieval: semantic + keyword → RRF fusion.

        Args:
            query: The user's question.
            top_k_semantic: How many results to fetch from the semantic retriever.
            top_k_bm25: How many results to fetch from the keyword retriever.
            top_k_final: If set, truncate the final fused list to this many results.
                         If None, all fused results are returned.

        Returns:
            A tuple of:
                - fused_results: the final ranked list of result dicts.
                - metrics: dict with retrieval_time_ms, num_semantic,
                           num_keyword, num_fused.
        """
        start_time: float = time.time()

        # 1. Retrieve from both sub-retrievers
        semantic_results = self.semantic_retriever.retrieve(query, top_k=top_k_semantic)
        keyword_results = self.keyword_retriever.retrieve(query, top_k=top_k_bm25)

        # 2. Fuse with RRF
        fused_results = self._reciprocal_rank_fusion(
            semantic_results, keyword_results
        )

        # 3. Deduplicate (RRF already handles merging, but just in case)
        fused_results = self._deduplicate(fused_results)

        # 4. Optionally truncate
        if top_k_final is not None:
            fused_results = fused_results[:top_k_final]

        elapsed_ms: float = (time.time() - start_time) * 1000.0

        metrics: Dict[str, Any] = {
            "retrieval_time_ms": round(elapsed_ms, 2),
            "num_semantic": len(semantic_results),
            "num_keyword": len(keyword_results),
            "num_fused": len(fused_results),
        }

        return fused_results, metrics
