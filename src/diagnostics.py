"""
Health Check and Diagnostics Module (Phase 7).
Provides readiness, liveness, and configuration diagnostics for production operations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from src.config import load_config
from src.vectorstore import FaissVectorStore


def check_api_key_configured() -> Dict[str, Any]:
    """Check if the required LLM API key is configured without exposing secrets."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return {
            "status": "missing",
            "message": "NVIDIA_API_KEY environment variable is not set. System can operate in retrieval-only mode.",
            "configured": False,
        }

    return {
        "status": "configured",
        "message": "NVIDIA_API_KEY is present.",
        "configured": True,
    }


def check_vector_store_health(persist_dir: str = "faiss_store") -> Dict[str, Any]:
    """Check availability, metadata alignment, and vector count of the active FAISS index."""
    p_path = Path(persist_dir)
    faiss_file = p_path / "faiss.index"
    meta_file = p_path / "metadata.pkl"
    model_file = p_path / "embedding_model.txt"

    if not p_path.exists() or not faiss_file.exists() or not meta_file.exists():
        return {
            "status": "not_found",
            "message": f"Vector store artifacts not found at '{persist_dir}'. Rebuild required.",
            "vector_count": 0,
            "aligned": False,
        }

    try:
        model_name = (
            model_file.read_text(encoding="utf-8").strip()
            if model_file.exists()
            else "unknown"
        )
        store = FaissVectorStore(persist_dir=persist_dir, embedding_model=model_name)
        store.load()

        vector_count = store.index.ntotal if store.index is not None else 0
        meta_count = len(store.metadata)
        aligned = vector_count == meta_count and vector_count > 0

        return {
            "status": "ready" if aligned else "misaligned",
            "message": f"Index loaded with {vector_count} vectors.",
            "vector_count": vector_count,
            "metadata_count": meta_count,
            "embedding_model": model_name,
            "aligned": aligned,
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to load vector store: {type(exc).__name__}",
            "vector_count": 0,
            "aligned": False,
        }


def check_filesystem_health(log_dir: str = "logs") -> Dict[str, Any]:
    """Verify that logging and runtime directories are writable."""
    l_path = Path(log_dir)
    try:
        l_path.mkdir(parents=True, exist_ok=True)
        test_file = l_path / ".health_check_tmp"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return {
            "status": "writable",
            "message": "Filesystem writable.",
            "writable": True,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Log directory '{log_dir}' is not writable: {type(exc).__name__}",
            "writable": False,
        }


def run_system_diagnostics(persist_dir: str = "faiss_store") -> Dict[str, Any]:
    """Execute complete system readiness diagnostic checks."""
    config = load_config()
    key_check = check_api_key_configured()
    store_check = check_vector_store_health(persist_dir or config.persist_dir)
    fs_check = check_filesystem_health(config.log_dir)

    # Determine overall health status
    if not store_check.get("aligned") or not fs_check.get("writable"):
        overall_status = "unready"
    elif not key_check.get("configured"):
        overall_status = "degraded"  # Retrieval works, LLM generation unavailable
    else:
        overall_status = "healthy"

    return {
        "overall_status": overall_status,
        "components": {
            "api_credentials": key_check,
            "vector_store": store_check,
            "filesystem": fs_check,
        },
        "config_summary": {
            "embedding_model": config.embedding_model,
            "llm_model": config.llm_model,
            "retrieval_k": config.retrieval_k,
            "rrf_k": config.rrf_k,
            "enable_expansion": config.enable_expansion,
            "enable_reranking": config.enable_reranking,
        },
    }
