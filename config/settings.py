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
    "GPT-OSS 120B": "openai/gpt-oss-120b",
    "GPT-OSS 20B": "openai/gpt-oss-20b",
    "Qwen 3.8 27B": "qwen/qwen3.8-27b",
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "DeepSeek R1 70B": "deepseek-r1-distill-llama-70b",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Allam 2 7B": "allam-2-7b",
}
DEFAULT_GROQ_MODEL: str = "openai/gpt-oss-120b"


def get_available_groq_models(api_key: str = "") -> dict[str, str]:
    """Retrieve dynamically available chat models for the provided Groq key.

    Queries the Groq API for models authorized for the given key.
    Falls back to curated GROQ_MODELS if lookup fails or key is missing.
    """
    effective_key = api_key or GROQ_API_KEY
    if not effective_key:
        return GROQ_MODELS

    try:
        from groq import Groq

        client = Groq(api_key=effective_key)
        models_data = client.models.list().data

        # Filter active chat models (exclude audio/whisper and guard models)
        chat_model_ids = [
            m.id
            for m in models_data
            if getattr(m, "active", True)
            and not m.id.startswith("whisper")
            and "guard" not in m.id
            and "orpheus" not in m.id  # Exclude models requiring custom terms
        ]

        if not chat_model_ids:
            return GROQ_MODELS

        dynamic_models: dict[str, str] = {}
        for mid in chat_model_ids:
            # Check known friendly names
            matched_name = None
            for p_name, p_id in GROQ_MODELS.items():
                if p_id == mid:
                    matched_name = p_name
                    break

            if matched_name:
                dynamic_models[matched_name] = mid
            else:
                clean_name = mid.replace("-", " ").replace("/", " - ").title()
                dynamic_models[clean_name] = mid

        return dynamic_models
    except Exception:
        return GROQ_MODELS

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
