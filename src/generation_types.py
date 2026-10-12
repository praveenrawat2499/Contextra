"""
Unified Response Schema and Citation Data Structures (Phase 6).
Provides typed, validated representations for grounded generation and citations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class Citation:
    """A citation reference linking a response claim to a retrieved chunk."""

    citation_index: int
    chunk_id: str
    source: str
    snippet: str
    verified: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Standardized response contract for all generation endpoints."""

    answer: str
    status: Literal["answered", "insufficient_evidence", "clarification_needed", "error"]
    citations: List[Citation] = field(default_factory=list)
    unverified_citations: List[int] = field(default_factory=list)
    trace_id: str = ""
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to serializable dictionary."""
        return asdict(self)

    def format_citations_markdown(self) -> str:
        """Render verified citations as markdown footnotes/links."""
        if not self.citations:
            return ""

        lines = ["\n\n**Sources & References:**"]
        for c in self.citations:
            source_label = c.source or "Document"
            lines.append(f"- **[{c.citation_index}]** `{source_label}` (Chunk: `{c.chunk_id[:10]}`) — *\"{c.snippet[:120]}...\"*")

        if self.unverified_citations:
            lines.append(f"- *(Note: citation reference(s) {self.unverified_citations} could not be verified against retrieved context)*")

        return "\n".join(lines)
