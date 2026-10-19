"""Text cleaning utilities for normalizing parsed document text.

Handles whitespace normalization, unicode cleanup, and special character removal
while preserving meaningful paragraph structure.
"""

import re
import unicodedata
from typing import Any, Dict, List


def clean_text(text: str) -> str:
    """Clean and normalize a text string.

    Steps:
        1. Normalize unicode characters (NFKD form).
        2. Replace multiple consecutive newlines with double newline (preserve paragraphs).
        3. Replace multiple spaces/tabs on the same line with a single space.
        4. Remove non-printable / control characters (except newline and tab).
        5. Strip leading/trailing whitespace.

    Args:
        text: The raw text to clean.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Step 1: Normalize unicode (e.g., fancy quotes → standard quotes)
    text = unicodedata.normalize("NFKD", text)

    # Step 2: Normalize horizontal whitespace (tabs, special spaces) to regular spaces, remove control chars
    text = re.sub(r"[^\S\n]", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Step 3: Collapse multiple blank lines into a single paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 4: Collapse multiple spaces on the same line
    text = re.sub(r" {2,}", " ", text)

    # Step 5: Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Step 6: Final strip
    text = text.strip()

    return text


def clean_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply text cleaning to a list of parsed document dictionaries.

    Each document dict is expected to have a 'text' key. Documents whose
    text becomes empty after cleaning are filtered out.

    Args:
        documents: List of dicts with at least a 'text' key.

    Returns:
        Cleaned list of document dicts (empty-text entries removed).
    """
    cleaned: List[Dict[str, Any]] = []

    for doc in documents:
        cleaned_text = clean_text(doc.get("text", ""))
        if cleaned_text:
            cleaned_doc = {**doc, "text": cleaned_text}
            cleaned.append(cleaned_doc)

    return cleaned
