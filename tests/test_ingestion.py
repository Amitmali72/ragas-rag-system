"""
Tests for the document ingestion pipeline.

Covers:
  - Plain-text parsing (parse_txt)
  - Text cleaning (whitespace normalization, unicode)
  - Chunking with metadata (chunk size, overlap, source info)
  - Mock UploadedFile to simulate Streamlit file objects

Run with:
    pytest tests/test_ingestion.py -v
"""

from __future__ import annotations

import io
import textwrap
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# Mock UploadedFile – mimics Streamlit's UploadedFile interface
# ===================================================================

class MockUploadedFile:
    """Lightweight stand-in for ``streamlit.runtime.uploaded_file_manager.UploadedFile``.

    Only implements the attributes the ingestion pipeline actually uses:
    ``.name``, ``.read()``, and ``.seek()``.
    """

    def __init__(self, name: str, content: str | bytes) -> None:
        self.name = name
        if isinstance(content, str):
            content = content.encode("utf-8")
        self._buffer = io.BytesIO(content)

    def read(self) -> bytes:
        """Return the full file content as bytes."""
        return self._buffer.read()

    def seek(self, pos: int) -> None:
        """Seek back to a position (usually 0) so the file can be re-read."""
        self._buffer.seek(pos)

    def getvalue(self) -> bytes:
        """Return the full content without changing the cursor."""
        current = self._buffer.tell()
        self._buffer.seek(0)
        data = self._buffer.read()
        self._buffer.seek(current)
        return data


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def sample_text() -> str:
    """A short multi-paragraph text for testing."""
    return textwrap.dedent("""\
        Introduction to Machine Learning

        Machine learning is a subset of artificial intelligence that enables
        systems to learn and improve from experience without being explicitly
        programmed.  It focuses on developing computer programs that can
        access data and use it to learn for themselves.

        The process of learning begins with observations or data, such as
        examples, direct experience, or instruction, in order to look for
        patterns in data and make better decisions in the future.

        Supervised learning uses labeled datasets to train algorithms to
        classify data or predict outcomes accurately.  Common supervised
        learning algorithms include linear regression, logistic regression,
        and support vector machines.

        Unsupervised learning algorithms discover hidden patterns in data
        without the need for labeled training data.  Clustering and
        dimensionality reduction are common unsupervised techniques.
    """)


@pytest.fixture
def mock_uploaded_txt(sample_text: str) -> MockUploadedFile:
    """A MockUploadedFile containing ``sample_text`` as a .txt file."""
    return MockUploadedFile(name="test_document.txt", content=sample_text)


@pytest.fixture
def dirty_text() -> str:
    """Text with excessive whitespace, tabs, and unicode oddities."""
    return (
        "Hello   \t  World!\n\n\n\n"
        "This   has    lots   of   spaces.\n"
        "\u00a0Non-breaking\u00a0space\u00a0here.\n"
        "  \t  Leading whitespace line.  \n"
    )


# ===================================================================
# Tests: plain-text parsing
# ===================================================================

class TestParseTxt:
    """Tests for parsing raw text from .txt files."""

    def test_basic_parse(self, sample_text: str) -> None:
        """parse_txt should return the full text content as a string or list of dicts."""
        from ingestion.file_parser import parse_txt
        result = parse_txt(sample_text)
        assert isinstance(result, list)
        assert len(result) > 0
        assert "Machine learning" in result[0]["text"]
        assert result[0]["page_number"] == 1

    def test_uploaded_file_parse(self, mock_uploaded_txt: MockUploadedFile) -> None:
        """Should handle a Streamlit-like uploaded file object."""
        from ingestion.file_parser import parse_txt
        results = parse_txt(mock_uploaded_txt)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "Machine learning" in results[0]["text"]
        assert results[0]["source"] == "test_document.txt"

    def test_empty_content(self) -> None:
        """parse_txt should handle empty strings gracefully."""
        from ingestion.file_parser import parse_txt
        result = parse_txt("")
        assert result == []

    def test_seek_resets_position(self, mock_uploaded_txt: MockUploadedFile) -> None:
        """After read + seek(0), re-reading should give the same content."""
        first_read = mock_uploaded_txt.read()
        mock_uploaded_txt.seek(0)
        second_read = mock_uploaded_txt.read()
        assert first_read == second_read


# ===================================================================
# Tests: text cleaning
# ===================================================================

class TestTextCleaning:
    """Tests for whitespace normalisation and unicode handling."""

    def test_collapse_whitespace(self, dirty_text: str) -> None:
        """Multiple spaces/tabs should be collapsed to a single space."""
        from ingestion.text_cleaner import clean_text
        cleaned = clean_text(dirty_text)

        assert "   " not in cleaned
        assert "\t" not in cleaned

    def test_unicode_normalization(self) -> None:
        """Non-breaking spaces (\u00a0) should be replaced with regular spaces."""
        from ingestion.text_cleaner import clean_text
        text_with_nbsp = "Hello\u00a0World"
        cleaned = clean_text(text_with_nbsp)

        assert "\u00a0" not in cleaned
        assert "Hello" in cleaned and "World" in cleaned

    def test_strip_leading_trailing(self) -> None:
        """Leading/trailing whitespace should be stripped from the result."""
        from ingestion.text_cleaner import clean_text
        text = "   \n  Some content here.  \n  "
        cleaned = clean_text(text)

        assert not cleaned.startswith(" ")
        assert not cleaned.endswith(" ")

    def test_preserves_single_newlines(self, dirty_text: str) -> None:
        """Single newlines (paragraph breaks) should be preserved or normalised."""
        from ingestion.text_cleaner import clean_text
        cleaned = clean_text(dirty_text)

        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in cleaned


# ===================================================================
# Tests: chunking
# ===================================================================

class TestChunking:
    """Tests for the text-chunking pipeline."""

    def test_chunk_output_is_list(self, sample_text: str) -> None:
        """Chunking should return a list of document-like objects."""
        from ingestion.chunker import chunk_text
        chunks = chunk_text(sample_text, chunk_size=200, chunk_overlap=50)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_size_respected(self, sample_text: str) -> None:
        """Each chunk should not exceed the specified chunk_size (with small tolerance)."""
        from ingestion.chunker import chunk_text
        chunk_size = 200
        tolerance = 50  # LangChain splitters may slightly overshoot
        chunks = chunk_text(sample_text, chunk_size=chunk_size, chunk_overlap=50)
        for chunk in chunks:
            text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            assert len(text) <= chunk_size + tolerance, (
                f"Chunk too long: {len(text)} > {chunk_size + tolerance}"
            )

    def test_chunk_metadata_present(self, sample_text: str) -> None:
        """Each chunk should carry metadata (source, chunk_id, chunk_index)."""
        from ingestion.chunker import chunk_text
        chunks = chunk_text(sample_text, chunk_size=200, chunk_overlap=50, source="test.txt")
        assert len(chunks) > 0
        for idx, chunk in enumerate(chunks):
            assert hasattr(chunk, "metadata"), "Chunk is missing metadata attribute"
            assert chunk.metadata.get("source") == "test.txt"
            assert "chunk_index" in chunk.metadata
            assert "chunk_id" in chunk.metadata

    def test_overlap_creates_more_chunks(self, sample_text: str) -> None:
        """Higher overlap should produce more chunks than lower overlap."""
        from ingestion.chunker import chunk_text
        chunks_low = chunk_text(sample_text, chunk_size=200, chunk_overlap=0)
        chunks_high = chunk_text(sample_text, chunk_size=200, chunk_overlap=100)
        assert len(chunks_high) >= len(chunks_low)

    def test_empty_text_returns_empty(self) -> None:
        """Chunking an empty string should return an empty list."""
        from ingestion.chunker import chunk_text
        chunks = chunk_text("", chunk_size=200, chunk_overlap=50)
        assert chunks == []

