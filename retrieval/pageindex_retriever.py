"""Structure-aware PageIndex Retriever.

Navigates documents hierarchically using document outlines, section titles,
and page boundaries. Distinguishable from BM25 and vector search by operating
on structural units (chapters, sections, pages) and mapping questions to specific
document regions before chunk extraction.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set
from langchain_core.documents import Document

from ingestion.structure_extractor import extract_document_structure, detect_heading


class PageIndexRetriever:
    """Retrieve documents using hierarchical, structure-aware PageIndex navigation."""

    def __init__(
        self,
        documents: List[Document],
        structure_index: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the PageIndex retriever with document chunks and optional structure.

        Args:
            documents: List of LangChain Document chunks.
            structure_index: Pre-extracted structural index. If omitted, constructed from chunks.
        """
        self.documents: List[Document] = documents

        if structure_index is not None:
            self.structure_index: Dict[str, Any] = structure_index
        else:
            # Build structure from document chunks
            parsed_representation = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page_number": doc.metadata.get("page_number", 1),
                    "text": doc.page_content,
                }
                for doc in documents
            ]
            self.structure_index = extract_document_structure(parsed_representation, documents)

    def _extract_page_references(self, query: str) -> Set[int]:
        """Extract explicit page numbers mentioned in query (e.g., 'page 3', 'p. 4')."""
        matches = re.findall(r"\b(?:page|pg|p\.?)\s*(\d+)\b", query, re.IGNORECASE)
        return {int(m) for m in matches}

    def _match_sections(self, query: str) -> List[Dict[str, Any]]:
        """Find sections matching the query terms or headings."""
        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))
        matched_sections: List[Dict[str, Any]] = []

        sources = self.structure_index.get("sources", {})
        for source_name, source_data in sources.items():
            for sec in source_data.get("sections", []):
                title = sec.get("title", "")
                title_lower = title.lower()
                title_words = set(re.findall(r"\w+", title_lower))

                # Exact or phrase match in title
                if title_lower in query_lower or query_lower in title_lower:
                    overlap_score = 1.0
                else:
                    overlap = len(query_words & title_words)
                    overlap_score = (overlap / len(title_words)) if title_words else 0.0

                if overlap_score > 0.2:
                    matched_sections.append({
                        "source": source_name,
                        "section": sec,
                        "score": overlap_score,
                    })

        # Sort descending by match score
        matched_sections.sort(key=lambda s: s["score"], reverse=True)
        return matched_sections

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks grounded in document structure and page hierarchy.

        Args:
            query: The user's query.
            top_k: Number of chunks to return.

        Returns:
            A list of result dicts conforming to the shared retrieval interface:
                - document: LangChain Document object
                - score: float in [0, 1]
                - rank: 1-based rank integer
                - retriever: literal "pageindex"
        """
        if not self.documents:
            return []

        page_refs = self._extract_page_references(query)
        matched_sections = self._match_sections(query)
        query_words = set(re.findall(r"\w+", query.lower()))

        candidate_scores: Dict[int, float] = {}

        # 1. Structural matching via explicit page references
        if page_refs:
            for idx, doc in enumerate(self.documents):
                doc_page = doc.metadata.get("page_number", 0)
                if doc_page in page_refs:
                    candidate_scores[idx] = candidate_scores.get(idx, 0.0) + 0.85

        # 2. Structural matching via section headings
        for match in matched_sections:
            sec = match["section"]
            sec_score = match["score"]
            for chunk_idx in sec.get("chunk_indices", []):
                if 0 <= chunk_idx < len(self.documents):
                    candidate_scores[chunk_idx] = candidate_scores.get(chunk_idx, 0.0) + (sec_score * 0.75)

        # 3. Contextual word overlap for chunks within target pages/sections
        for idx, doc in enumerate(self.documents):
            chunk_words = set(re.findall(r"\w+", doc.page_content.lower()))
            overlap = len(query_words & chunk_words)
            lexical_ratio = (overlap / len(query_words)) if query_words else 0.0

            if idx in candidate_scores:
                # Boost structurally targeted chunks with internal lexical alignment
                candidate_scores[idx] += lexical_ratio * 0.3
            elif not page_refs and not matched_sections:
                # If no explicit structure was identified in the query, score by page structure density
                if lexical_ratio > 0.1:
                    candidate_scores[idx] = lexical_ratio * 0.6

        if not candidate_scores:
            # Fallback: take first chunks if totally zero matches
            sorted_indices = list(range(min(top_k, len(self.documents))))
        else:
            sorted_indices = sorted(
                candidate_scores.keys(),
                key=lambda i: candidate_scores[i],
                reverse=True,
            )[:top_k]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            raw_score = candidate_scores.get(idx, 0.1)
            # Normalize score into [0, 1]
            normalized_score = min(1.0, max(0.05, round(raw_score / 1.5, 4)))

            doc = self.documents[idx]
            results.append({
                "document": doc,
                "score": normalized_score,
                "rank": rank,
                "retriever": "pageindex",
            })

        return results
