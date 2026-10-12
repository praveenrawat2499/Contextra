"""
Standardized types and data classes for retrieval operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RetrievalResult:
    """Standardized retrieval result representation across dense, sparse, and hybrid retrievers."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    retriever: str
    rank: int
    score: float = 0.0
    score_type: str = "rank"
    provenance: Optional[Dict[str, Any]] = None
