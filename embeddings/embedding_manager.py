"""Embedding and FAISS vector store management.

Provides helpers to create, save, load, and extend a FAISS index backed
by HuggingFace sentence-transformer embeddings.  All paths are relative
to the project root unless an absolute path is supplied.
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL, FAISS_INDEX_PATH


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a HuggingFace embedding model instance.

    Uses the model name defined in config.settings.EMBEDDING_MODEL
    (default: BAAI/bge-small-en-v1.5, 384-dim).

    Returns:
        A LangChain HuggingFaceEmbeddings instance ready for encoding.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},       # safe default; change to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedding_model


def create_faiss_index(documents: List[Document]) -> FAISS:
    """Create a new FAISS index from a list of LangChain Document objects.

    Args:
        documents: List of LangChain Document objects (with page_content & metadata).

    Returns:
        A FAISS vector store instance.
    """
    embedding_model = get_embedding_model()
    faiss_index = FAISS.from_documents(documents, embedding_model)
    return faiss_index


def save_faiss_index(index: FAISS, path: str = FAISS_INDEX_PATH) -> None:
    """Persist a FAISS index to disk.

    Creates the target directory if it doesn't exist.

    Args:
        index: The FAISS vector store to save.
        path: Directory path where the index files will be written.
    """
    os.makedirs(path, exist_ok=True)
    index.save_local(path)


def load_faiss_index(path: str = FAISS_INDEX_PATH) -> FAISS:
    """Load a previously saved FAISS index from disk.

    The same embedding model used during creation is required to
    deserialize the index correctly.

    Args:
        path: Directory path containing the saved index files.

    Returns:
        A FAISS vector store instance.

    Raises:
        FileNotFoundError: If the index directory does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FAISS index not found at '{path}'. "
            "Please ingest documents first to create an index."
        )

    embedding_model = get_embedding_model()
    faiss_index = FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True,  # required by newer LangChain versions
    )
    return faiss_index


def add_documents_to_index(
    index: FAISS, documents: List[Document]
) -> FAISS:
    """Add new documents to an existing FAISS index.

    The documents are embedded with the same model and merged into the
    existing index in-place.

    Args:
        index: The existing FAISS vector store.
        documents: New LangChain Document objects to add.

    Returns:
        The updated FAISS vector store (same object, mutated in-place).
    """
    index.add_documents(documents)
    return index
