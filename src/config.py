"""
Centralized Configuration Module (Phase 7).
Provides validated application settings, timeouts, feature flags, and environment defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration."""

    persist_dir: str = "faiss_store"
    data_dir: str = "data"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    llm_model: str = "z-ai/glm-5.3-flash"
    retrieval_k: int = 10
    rrf_k: int = 60
    enable_expansion: bool = False
    enable_reranking: bool = False
    reranker_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    max_query_chars: int = 1000
    request_timeout_s: int = 60
    log_dir: str = "logs"
    log_file: str = "logs/rag.log"


def load_config() -> AppConfig:
    """Load configuration from environment variables with safe defaults and validation."""
    persist_dir = os.getenv("PERSIST_DIR", "faiss_store")
    data_dir = os.getenv("DATA_DIR", "data")
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    llm_model = os.getenv("LLM_MODEL", "z-ai/glm-5.3-flash")

    try:
        retrieval_k = max(1, int(os.getenv("RETRIEVAL_K", "10")))
    except ValueError:
        retrieval_k = 10

    try:
        rrf_k = max(1, int(os.getenv("RRF_K", "60")))
    except ValueError:
        rrf_k = 60

    enable_expansion = os.getenv("ENABLE_EXPANSION", "false").lower() in ("true", "1", "yes")
    enable_reranking = os.getenv("ENABLE_RERANKING", "false").lower() in ("true", "1", "yes")
    reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-TinyBERT-L-2-v2")

    try:
        max_query_chars = max(100, int(os.getenv("MAX_QUERY_CHARS", "1000")))
    except ValueError:
        max_query_chars = 1000

    try:
        request_timeout_s = max(5, int(os.getenv("REQUEST_TIMEOUT_S", "60")))
    except ValueError:
        request_timeout_s = 60

    log_dir = os.getenv("LOG_DIR", "logs")
    log_file = os.getenv("LOG_FILE", "logs/rag.log")

    return AppConfig(
        persist_dir=persist_dir,
        data_dir=data_dir,
        embedding_model=embedding_model,
        llm_model=llm_model,
        retrieval_k=retrieval_k,
        rrf_k=rrf_k,
        enable_expansion=enable_expansion,
        enable_reranking=enable_reranking,
        reranker_model=reranker_model,
        max_query_chars=max_query_chars,
        request_timeout_s=request_timeout_s,
        log_dir=log_dir,
        log_file=log_file,
    )
