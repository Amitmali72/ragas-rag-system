"""Tests for structure-aware PageIndex retrieval.

Covers:
  - Document outline and heading detection
  - Page-tree mapping and chunk linkage
  - Explicit page reference retrieval
  - Section title / structural keyword retrieval
  - Shared retrieval interface compliance
  - Unstructured document fallback
"""

import pytest
from langchain_core.documents import Document

from ingestion.structure_extractor import detect_heading, extract_document_structure
from retrieval.pageindex_retriever import PageIndexRetriever


@pytest.fixture
def structured_documents() -> list[Document]:
    """Sample structured corpus spanning multiple pages and sections."""
    return [
        Document(
            page_content="1. Executive Summary\nRAGNEXUS is an adaptive knowledge retrieval system.",
            metadata={"source": "report.pdf", "page_number": 1, "chunk_id": "report_p1_c0"},
        ),
        Document(
            page_content="The primary objective is eliminating hallucinations through strict grounding.",
            metadata={"source": "report.pdf", "page_number": 1, "chunk_id": "report_p1_c1"},
        ),
        Document(
            page_content="2. System Architecture\nThe architecture includes dense FAISS and sparse BM25 indices.",
            metadata={"source": "report.pdf", "page_number": 2, "chunk_id": "report_p2_c0"},
        ),
        Document(
            page_content="Section 3: Evaluation Protocol\nRAGAS metrics are used for benchmarking.",
            metadata={"source": "report.pdf", "page_number": 3, "chunk_id": "report_p3_c0"},
        ),
    ]


class TestStructureExtractor:
    """Tests for outline and heading extraction."""

    def test_detect_numbered_heading(self) -> None:
        assert detect_heading("1. Introduction") == "1. Introduction"
        assert detect_heading("2.3 Methodology") == "2.3 Methodology"

    def test_detect_section_keyword_heading(self) -> None:
        assert detect_heading("Section 4: Results") == "Section 4: Results"
        assert detect_heading("Chapter 2. Related Work") == "Chapter 2. Related Work"

    def test_detect_markdown_heading(self) -> None:
        assert detect_heading("## System Design") == "System Design"

    def test_ignore_normal_sentences(self) -> None:
        assert detect_heading("This is a normal paragraph discussing the system.") is None

    def test_extract_structure_from_documents(self, structured_documents: list[Document]) -> None:
        parsed = [
            {"source": d.metadata["source"], "page_number": d.metadata["page_number"], "text": d.page_content}
            for d in structured_documents
        ]
        structure = extract_document_structure(parsed, structured_documents)

        assert "sources" in structure
        assert "report.pdf" in structure["sources"]
        source_data = structure["sources"]["report.pdf"]
        assert len(source_data["pages"]) == 3
        assert len(source_data["sections"]) >= 2


class TestPageIndexRetriever:
    """Tests for PageIndexRetriever behavior and interface adherence."""

    def test_retriever_initialization(self, structured_documents: list[Document]) -> None:
        retriever = PageIndexRetriever(structured_documents)
        assert len(retriever.documents) == len(structured_documents)
        assert retriever.structure_index["total_pages"] == 3

    def test_page_reference_targeting(self, structured_documents: list[Document]) -> None:
        retriever = PageIndexRetriever(structured_documents)
        results = retriever.retrieve("Show me what is discussed on page 2", top_k=2)

        assert len(results) > 0
        top_result = results[0]
        assert top_result["document"].metadata["page_number"] == 2
        assert "System Architecture" in top_result["document"].page_content
        assert top_result["retriever"] == "pageindex"

    def test_section_heading_targeting(self, structured_documents: list[Document]) -> None:
        retriever = PageIndexRetriever(structured_documents)
        results = retriever.retrieve("Tell me about the Executive Summary", top_k=2)

        assert len(results) > 0
        top_doc = results[0]["document"]
        assert "Executive Summary" in top_doc.page_content or top_doc.metadata["page_number"] == 1

    def test_interface_compliance(self, structured_documents: list[Document]) -> None:
        """Every result must have document, score in [0, 1], rank, and retriever='pageindex'."""
        retriever = PageIndexRetriever(structured_documents)
        results = retriever.retrieve("Evaluation Protocol", top_k=3)

        assert len(results) > 0
        for item in results:
            assert "document" in item
            assert "score" in item
            assert "rank" in item
            assert item["retriever"] == "pageindex"
            assert 0.0 <= item["score"] <= 1.0

    def test_unstructured_document_graceful_handling(self) -> None:
        """Documents without headings should still return valid results without errors."""
        unstructured = [
            Document(page_content="Just plain text without headings.", metadata={"source": "plain.txt", "page_number": 1}),
            Document(page_content="Another sentence without headings.", metadata={"source": "plain.txt", "page_number": 1}),
        ]
        retriever = PageIndexRetriever(unstructured)
        results = retriever.retrieve("plain text", top_k=2)

        assert len(results) > 0
        assert results[0]["retriever"] == "pageindex"

    def test_empty_documents_returns_empty(self) -> None:
        retriever = PageIndexRetriever([])
        results = retriever.retrieve("test query", top_k=5)
        assert results == []
