"""Central configuration for Knowledge Copilot - Hybrid RAG System.

This module contains all configurable parameters for the RAG pipeline,
including document processing, embedding, retrieval, LLM, and evaluation settings.
All settings can be overridden via environment variables where applicable.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# --- Document Processing ---
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
SUPPORTED_FILE_TYPES: list[str] = ["pdf", "docx", "txt"]

# --- Embedding Model ---
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION: int = 384

# --- Vector Store ---
FAISS_INDEX_PATH: str = "vector_store"

# --- Retrieval ---
TOP_K_SEMANTIC: int = 5
TOP_K_BM25: int = 5
RRF_K: int = 60  # Reciprocal Rank Fusion constant

# --- Groq LLM ---
GROQ_API_KEY: str = ""
try:
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not GROQ_API_KEY:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODELS: dict[str, str] = {
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "DeepSeek R1 70B": "deepseek-r1-distill-llama-70b",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
}
DEFAULT_GROQ_MODEL: str = "llama-3.3-70b-versatile"

# --- Generation ---
MAX_CONTEXT_LENGTH: int = 4000
TEMPERATURE: float = 0.0

# --- System Prompt ---
SYSTEM_PROMPT: str = """You are Knowledge Copilot, an intelligent document assistant.
You MUST follow these rules strictly:

1. Answer ONLY based on the provided context from uploaded documents.
2. Do NOT invent, assume, or hallucinate any information.
3. If the answer is not found in the context, respond exactly with:
   "Information not found in uploaded documents."
4. Always cite the source document name and page number when providing information.
5. Be concise, accurate, and helpful.
6. If the context contains partial information, clearly state what was found and what is missing."""

# --- Evaluation ---
SAMPLE_EVAL_PATH: str = "data/sample/sample_qa.json"
