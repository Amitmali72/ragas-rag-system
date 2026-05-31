"""Groq LLM integration for answer generation.

Provides helpers to initialise a ChatGroq instance, format retrieved context
into a prompt, and generate answers with or without RAG context.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import (
    DEFAULT_GROQ_MODEL,
    GROQ_API_KEY,
    MAX_CONTEXT_LENGTH,
    SYSTEM_PROMPT,
)


def get_groq_llm(
    model_name: str = DEFAULT_GROQ_MODEL,
    api_key: Optional[str] = None,
) -> ChatGroq:
    """Create and return a ChatGroq LLM instance.

    Args:
        model_name: The Groq model identifier (e.g., "llama-3.3-70b-versatile").
        api_key: Groq API key.  Falls back to the key in config.settings.

    Returns:
        A LangChain ChatGroq instance ready for inference.
    """
    resolved_key: str = api_key or GROQ_API_KEY

    if not resolved_key:
        raise ValueError(
            "Groq API key is not set. Please set the GROQ_API_KEY environment "
            "variable or pass it explicitly."
        )

    llm = ChatGroq(
        model_name=model_name,
        api_key=resolved_key,
        temperature=0,
    )
    return llm


def format_context(chunks: List[Any]) -> str:
    """Format retrieved chunks into a numbered context string.

    Supports chunks as either:
      - A list of LangChain Document objects
      - A list of dicts with a 'document' key (which is a Document)
      - A list of dicts with 'text'/'page_content' and 'metadata'/'source' keys

    Args:
        chunks: List of retrieved chunks.

    Returns:
        A single string containing all chunks, truncated to MAX_CONTEXT_LENGTH.
    """
    context_parts: List[str] = []

    for i, chunk in enumerate(chunks, start=1):
        doc = None
        text = ""
        source = "Unknown"
        page = 0

        if hasattr(chunk, "page_content"):
            # It's a Document object
            doc = chunk
        elif isinstance(chunk, dict):
            if "document" in chunk:
                # It's a dict containing a Document object
                doc = chunk["document"]
            elif "page_content" in chunk or "text" in chunk:
                # It's a raw dictionary representation
                text = chunk.get("page_content", chunk.get("text", ""))
                metadata = chunk.get("metadata", {})
                source = metadata.get("source", chunk.get("source", "Unknown"))
                page = metadata.get("page_number", chunk.get("page", 0))

        if doc is not None:
            text = doc.page_content
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page_number", 0)

        context_parts.append(
            f"[Chunk {i}] (Source: {source}, Page: {page})\n{text}"
        )

    full_context: str = "\n\n".join(context_parts)

    # Truncate if the context exceeds the maximum allowed length
    if len(full_context) > MAX_CONTEXT_LENGTH:
        full_context = full_context[:MAX_CONTEXT_LENGTH] + "\n...[context truncated]"

    return full_context


def generate_answer(
    query: str,
    context_chunks: List[Any],
    llm: ChatGroq,
) -> Tuple[str, float]:
    """Generate an answer grounded in retrieved document context.

    Constructs a prompt with the system instructions and formatted context,
    then calls the LLM.

    Args:
        query: The user's question.
        context_chunks: Retrieved chunks (output of a retriever's .retrieve()).
        llm: A ChatGroq instance.

    Returns:
        A tuple of (answer_text, generation_time_ms).
    """
    context_str: str = format_context(context_chunks)

    human_prompt: str = (
        f"Context from uploaded documents:\n"
        f"---\n{context_str}\n---\n\n"
        f"Question: {query}\n\n"
        f"Provide a detailed answer based ONLY on the above context. "
        f"Cite the source document and page number for every claim."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    start_time: float = time.time()
    response = llm.invoke(messages)
    elapsed_ms: float = (time.time() - start_time) * 1000.0

    answer: str = response.content
    return answer, round(elapsed_ms, 2)


def generate_direct_answer(
    query: str,
    llm: ChatGroq,
) -> Tuple[str, float]:
    """Generate an answer directly from the LLM without RAG context.

    Useful as a baseline to compare against RAG-augmented answers.

    Args:
        query: The user's question.
        llm: A ChatGroq instance.

    Returns:
        A tuple of (answer_text, generation_time_ms).
    """
    system_prompt: str = (
        "You are a helpful assistant. "
        "Answer the question to the best of your knowledge."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    start_time: float = time.time()
    response = llm.invoke(messages)
    elapsed_ms: float = (time.time() - start_time) * 1000.0

    answer: str = response.content
    return answer, round(elapsed_ms, 2)
