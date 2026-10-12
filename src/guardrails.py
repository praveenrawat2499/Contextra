"""
Guardrails and Security Module (Phase 6).
Provides input sanitization, secret redaction, prompt injection quarantine, and access control.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from src.retrieval_types import RetrievalResult


# Regex patterns for credential and secret redaction
SECRET_PATTERNS = [
    re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9\._\-]{20,}", re.IGNORECASE),
    re.compile(r"ghp_[A-Za-z0-9]{20,}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{16,})['\"]?", re.IGNORECASE),
]

# Patterns attempting prompt injection or system instruction overrides
INJECTION_MARKERS = [
    re.compile(r"<\s*\|\s*(?:im_start|im_end|system|user|assistant)\s*\|?\s*>", re.IGNORECASE),
    re.compile(r"\[\s*(?:system|instruction|developer|admin)\s*(?:instruction|mode|prompt)?\s*\]", re.IGNORECASE),
    re.compile(r"(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|rules|prompts)", re.IGNORECASE),
]


def redact_secrets(text: str) -> str:
    """Redact sensitive API keys, tokens, and passwords from logs, queries, and errors."""
    if not text:
        return text

    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


def sanitize_query(query: str, max_chars: int = 1000) -> tuple[str, Optional[str]]:
    """Validate and sanitize user query.

    Returns:
        (sanitized_query, error_message_if_invalid)
    """
    if not query or not query.strip():
        return "", "Query cannot be empty."

    clean = " ".join(query.split()).strip()

    if len(clean) > max_chars:
        return "", f"Query exceeds maximum allowed length of {max_chars} characters."

    # Redact any accidental secret strings in the query itself
    clean = redact_secrets(clean)
    return clean, None


def sanitize_context_chunk(chunk_text: str) -> str:
    """Neutralize adversarial prompt injection tokens in untrusted retrieved documents."""
    if not chunk_text:
        return ""

    sanitized = chunk_text
    for marker in INJECTION_MARKERS:
        sanitized = marker.sub("[UNTRUSTED_CONTENT_FILTERED]", sanitized)

    return sanitized


def filter_authorized_documents(
    candidates: List[RetrievalResult],
    allowed_sources: Optional[Set[str]] = None,
    tenant_id: Optional[str] = None,
) -> List[RetrievalResult]:
    """Enforce access control and multi-tenant security in application code before LLM context construction."""
    if allowed_sources is None and tenant_id is None:
        return candidates

    filtered = []
    for c in candidates:
        meta = c.metadata or {}

        # Tenant filter
        if tenant_id is not None:
            doc_tenant = meta.get("tenant_id")
            if doc_tenant and doc_tenant != tenant_id:
                continue

        # Allowed source documents filter
        if allowed_sources is not None:
            source = meta.get("source")
            if source and source not in allowed_sources:
                continue

        filtered.append(c)

    return filtered
