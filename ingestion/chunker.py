"""Document chunking module using LangChain's RecursiveCharacterTextSplitter.

Splits cleaned documents into overlapping chunks suitable for embedding and
retrieval. Each chunk carries metadata (source, page, chunk_id, chunk_index)
for downstream citation and traceability.
"""

from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def create_chunks(
    documents: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split parsed documents into overlapping chunks.

    Uses LangChain's RecursiveCharacterTextSplitter which tries to split
    on paragraph boundaries first, then sentences, then words, to keep
    semantically coherent pieces together.

    Args:
        documents: List of dicts with keys 'text', 'source', 'page_number'.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of LangChain Document objects with metadata:
            - source: original file name
            - page_number: page the text came from
            - chunk_id: human-readable identifier  (e.g., "report.pdf_p1_c0")
            - chunk_index: integer index of the chunk within the full corpus
    """
    # Configure the splitter with sensible separator hierarchy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # paragraph → sentence → word
    )

    all_chunks: List[Document] = []
    global_chunk_index: int = 0

    for doc in documents:
        text: str = doc.get("text", "")
        source: str = doc.get("source", "unknown")
        page_number: int = doc.get("page_number", 1)

        if not text.strip():
            continue

        # Split text into raw string chunks
        splits: List[str] = text_splitter.split_text(text)

        for local_idx, chunk_text in enumerate(splits):
            # Build a human-readable chunk ID: "filename_p<page>_c<local_index>"
            chunk_id = f"{source}_p{page_number}_c{local_idx}"

            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": source,
                    "page_number": page_number,
                    "chunk_id": chunk_id,
                    "chunk_index": global_chunk_index,
                },
            )
            all_chunks.append(chunk_doc)
            global_chunk_index += 1

    return all_chunks


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    source: str = "unknown",
    page_number: int = 1,
) -> List[Document]:
    """Helper to split a raw text string into Document chunks.

    Args:
        text: Raw text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.
        source: Source identifier / file name.
        page_number: Page number for attribution.

    Returns:
        List of LangChain Document objects with chunk metadata.
    """
    if not text or not text.strip():
        return []
    return create_chunks(
        [{"text": text, "source": source, "page_number": page_number}],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
