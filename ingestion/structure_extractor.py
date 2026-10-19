"""Structure extractor module for structure-aware PageIndex retrieval.

Extracts hierarchical outlines, section headers, and page-level metadata from
parsed documents. Enables navigation by structural boundaries (headings, sections,
pages) rather than treating text as a flat sequence of chunks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document


# Common patterns for section headings across PDF/DOCX/TXT
HEADING_PATTERNS = [
    re.compile(r"^(?:section|chapter|part)\s+[\dA-ZIVX]+[:.\s-].*", re.IGNORECASE),
    re.compile(r"^\d+(?:\.\d+)*[.:]?\s+[A-Z].*"),             # e.g., "1. Introduction", "2.1 Background"
    re.compile(r"^#{1,6}\s+.+"),                             # Markdown headers: "# Heading"
    re.compile(r"^[A-Z][A-Z0-9\s,\-:]{3,60}$"),             # ALL CAPS headings
]


def detect_heading(line: str) -> Optional[str]:
    """Check if a text line appears to be a section or chapter heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return None

    for pattern in HEADING_PATTERNS:
        if pattern.match(stripped):
            # Clean markdown symbols if present
            clean_heading = re.sub(r"^#{1,6}\s+", "", stripped)
            return clean_heading.strip()

    return None


def extract_document_structure(
    parsed_docs: List[Dict[str, Any]],
    chunks: Optional[List[Document]] = None,
) -> Dict[str, Any]:
    """Build a structural tree / PageIndex mapping from parsed documents.

    Args:
        parsed_docs: List of parsed document dictionaries (text, source, page_number).
        chunks: Optional list of LangChain Document chunks to map directly to sections.

    Returns:
        A structural index dict:
            {
                "sources": {
                    "filename.pdf": {
                        "pages": {
                            1: {
                                "headings": ["1. Executive Summary"],
                                "text_preview": "...",
                                "chunk_indices": [0, 1]
                            }
                        },
                        "sections": [
                            {
                                "title": "1. Executive Summary",
                                "page_number": 1,
                                "chunk_indices": [0, 1]
                            }
                        ]
                    }
                },
                "total_sections": int,
                "total_pages": int,
            }
    """
    structure_by_source: Dict[str, Dict[str, Any]] = {}
    total_sections = 0
    total_pages = 0

    for doc in parsed_docs:
        source = doc.get("source", "unknown")
        page_num = doc.get("page_number", 1)
        text = doc.get("text", "")

        if source not in structure_by_source:
            structure_by_source[source] = {
                "pages": {},
                "sections": [],
            }

        if page_num not in structure_by_source[source]["pages"]:
            total_pages += 1
            structure_by_source[source]["pages"][page_num] = {
                "headings": [],
                "text_preview": text[:200].strip(),
                "chunk_indices": [],
            }

        # Detect any headings within this page's text
        for line in text.splitlines():
            heading = detect_heading(line)
            if heading and heading not in structure_by_source[source]["pages"][page_num]["headings"]:
                structure_by_source[source]["pages"][page_num]["headings"].append(heading)
                structure_by_source[source]["sections"].append({
                    "title": heading,
                    "page_number": page_num,
                    "chunk_indices": [],
                })
                total_sections += 1

    # Map chunks to structure if chunks are provided
    if chunks:
        for idx, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            page_num = chunk.metadata.get("page_number", 1)
            chunk_content = chunk.page_content.lower()

            if source in structure_by_source and page_num in structure_by_source[source]["pages"]:
                structure_by_source[source]["pages"][page_num]["chunk_indices"].append(idx)

                # Check if chunk belongs under a specific section in this page
                for sec in structure_by_source[source]["sections"]:
                    if sec["page_number"] == page_num or sec["title"].lower() in chunk_content:
                        if idx not in sec["chunk_indices"]:
                            sec["chunk_indices"].append(idx)

    return {
        "sources": structure_by_source,
        "total_sections": total_sections,
        "total_pages": total_pages,
    }
