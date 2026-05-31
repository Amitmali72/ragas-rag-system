"""Citation formatting utilities for presenting source attributions.

Converts raw retrieval results into user-friendly citation objects
that can be rendered as Markdown in the Streamlit UI.
"""

from typing import Any, Dict, List


def format_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract and format citation metadata from retrieved chunks.

    Args:
        chunks: List of result dicts from a retriever.  Each dict must
                contain a 'document' key (LangChain Document with metadata)
                and optionally a 'retrievers' key (list of retriever names).

    Returns:
        List of citation dicts, each with:
            - source: original file name
            - page: page number
            - chunk_id: unique chunk identifier
            - retrievers: list of retrievers that found this chunk
            - preview: first 100 characters of the chunk text
    """
    citations: List[Dict[str, Any]] = []

    for chunk in chunks:
        doc = chunk["document"]
        metadata: dict = doc.metadata

        citation: Dict[str, Any] = {
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page_number", 0),
            "page_number": metadata.get("page_number", 0),  # Alias for compatibility
            "chunk_id": metadata.get("chunk_id", "N/A"),
            "retrievers": chunk.get("retrievers", [chunk.get("retriever", "unknown")]),
            "preview": doc.page_content[:100].strip(),
            "text": doc.page_content,  # Alias for compatibility
        }
        citations.append(citation)

    return citations


def format_citations_markdown(citations: List[Dict[str, Any]]) -> str:
    """Render a list of citations as a Markdown string.

    Uses emoji icons for visual scanning:
        📄 source file name
        📃 page number

    Args:
        citations: List of citation dicts (output of format_citations).

    Returns:
        Markdown-formatted string listing all citations.
    """
    if not citations:
        return "_No citations available._"

    lines: List[str] = ["#### 📚 Sources", ""]

    for i, cit in enumerate(citations, start=1):
        retrievers_str = ", ".join(cit.get("retrievers", []))
        preview = cit.get("preview", "")

        lines.append(
            f"**{i}.** 📄 **{cit['source']}** · 📃 Page {cit['page']}"
        )
        lines.append(f"   _Retrievers: {retrievers_str}_")
        lines.append(f"   > {preview}…")
        lines.append("")

    return "\n".join(lines)


def group_citations_by_source(
    citations: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group citations by their source document name.

    Useful for displaying a per-document breakdown in the UI.

    Args:
        citations: List of citation dicts (output of format_citations).

    Returns:
        Dict mapping source file names to lists of citation dicts.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for cit in citations:
        source: str = cit.get("source", "Unknown")
        grouped.setdefault(source, []).append(cit)

    return grouped
