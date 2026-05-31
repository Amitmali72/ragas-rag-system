"""File parser module for extracting text from PDF, DOCX, and TXT files.

Supports Streamlit UploadedFile objects (which expose .name and .read() methods).
Each parser returns a list of dictionaries with keys: text, source, page_number.
"""

import io
from typing import Any, Dict, List

from PyPDF2 import PdfReader
from docx import Document


def parse_pdf(file: Any) -> List[Dict[str, Any]]:
    """Parse a PDF file and extract text page by page.

    Args:
        file: A Streamlit UploadedFile or file-like object with .name and .read().

    Returns:
        List of dicts, each containing 'text', 'source', and 'page_number'.
    """
    documents: List[Dict[str, Any]] = []

    try:
        # Read file bytes and wrap in BytesIO for PyPDF2
        pdf_bytes = file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))

        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():  # Only keep pages with actual content
                documents.append({
                    "text": text,
                    "source": file.name,
                    "page_number": page_num,
                })

    except Exception as e:
        raise ValueError(f"Error parsing PDF file '{file.name}': {e}") from e

    return documents


def parse_docx(file: Any) -> List[Dict[str, Any]]:
    """Parse a DOCX file and extract text from paragraphs.

    Paragraphs are grouped into synthetic "pages" of ~40 paragraphs each
    to approximate page boundaries (DOCX files don't have true page info).

    Args:
        file: A Streamlit UploadedFile or file-like object with .name and .read().

    Returns:
        List of dicts, each containing 'text', 'source', and 'page_number'.
    """
    documents: List[Dict[str, Any]] = []
    paragraphs_per_page = 40  # Approximate paragraphs per page

    try:
        # Read file bytes and wrap in BytesIO for python-docx
        docx_bytes = file.read()
        doc = Document(io.BytesIO(docx_bytes))

        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        if not paragraphs:
            return documents

        # Group paragraphs into synthetic pages
        for i in range(0, len(paragraphs), paragraphs_per_page):
            page_paragraphs = paragraphs[i : i + paragraphs_per_page]
            page_text = "\n".join(page_paragraphs)
            page_number = (i // paragraphs_per_page) + 1

            documents.append({
                "text": page_text,
                "source": file.name,
                "page_number": page_number,
            })

    except Exception as e:
        raise ValueError(f"Error parsing DOCX file '{file.name}': {e}") from e

    return documents


def parse_txt(file: Any) -> List[Dict[str, Any]]:
    """Parse a plain text file.

    The entire file content is treated as a single page.

    Args:
        file: A Streamlit UploadedFile or file-like object with .name and .read().

    Returns:
        List of dicts with a single entry containing 'text', 'source', and 'page_number'.
    """
    documents: List[Dict[str, Any]] = []

    try:
        raw_bytes = file.read()
        text = raw_bytes.decode("utf-8", errors="replace")

        if text.strip():
            documents.append({
                "text": text,
                "source": file.name,
                "page_number": 1,
            })

    except Exception as e:
        raise ValueError(f"Error parsing TXT file '{file.name}': {e}") from e

    return documents


def parse_file(file: Any) -> List[Dict[str, Any]]:
    """Route a file to the appropriate parser based on its extension.

    Args:
        file: A Streamlit UploadedFile or file-like object with .name and .read().

    Returns:
        List of dicts, each containing 'text', 'source', and 'page_number'.

    Raises:
        ValueError: If the file extension is not supported.
    """
    filename: str = file.name.lower()

    if filename.endswith(".pdf"):
        return parse_pdf(file)
    elif filename.endswith(".docx"):
        return parse_docx(file)
    elif filename.endswith(".txt"):
        return parse_txt(file)
    else:
        supported = ", ".join(["pdf", "docx", "txt"])
        raise ValueError(
            f"Unsupported file type: '{filename}'. Supported types: {supported}"
        )
