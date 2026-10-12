"""
Citation Validator and Grounded Generation Verification (Phase 6).
Extracts, validates, and cross-references citation markers against retrieved evidence chunks.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from src.generation_types import Citation, RAGResponse
from src.retrieval_types import RetrievalResult


CITATION_REGEX = re.compile(r"\[(?:Context|cite|ref)?\s*(\d+)\]", re.IGNORECASE)

ABSTENTION_PHRASES = [
    "i cannot find",
    "i couldn't find",
    "the retrieved documents do not contain",
    "the retrieved context does not contain",
    "the retrieved context does not",
    "does not mention",
    "does not contain",
    "not mentioned in the context",
    "not mentioned in the documents",
    "insufficient evidence",
    "no relevant documents",
    "i don't have enough information",
    "the documents provided do not",
    "the context does not provide",
    "can't share",
    "cannot share",
    "unable to reveal",
    "can't help with that request",
    "cannot help with that request",
]


def extract_citation_indices(text: str) -> List[int]:
    """Extract distinct 1-indexed citation reference numbers found in generation text."""
    if not text:
        return []

    found = []
    seen = set()
    for match in CITATION_REGEX.finditer(text):
        try:
            idx = int(match.group(1))
            if idx not in seen:
                seen.add(idx)
                found.append(idx)
        except ValueError:
            continue
    return found


def is_abstention_response(text: str) -> bool:
    """Detect whether generated text conveys explicit abstention or insufficient evidence."""
    if not text:
        return True

    lower = text.casefold()
    return any(phrase in lower for phrase in ABSTENTION_PHRASES)


def validate_and_build_response(
    answer: str,
    evidence_chunks: List[RetrievalResult],
    trace_id: str = "",
    model: str = "",
) -> RAGResponse:
    """Validate citations against retrieved chunks and return a typed RAGResponse.

    Guarantees:
    - Every cited index must correspond to an actual chunk in evidence_chunks (1-indexed).
    - If cited index is out of bounds, it is recorded as unverified.
    - Verified citations extract actual chunk_id, source filename, and snippet from metadata.
    - Explicit abstention automatically marks status as 'insufficient_evidence'.
    """
    clean_answer = (answer or "").strip()
    if not clean_answer:
        return RAGResponse(
            answer="The model returned an empty response.",
            status="error",
            trace_id=trace_id,
            model=model,
        )

    # Check for explicit abstention
    if is_abstention_response(clean_answer):
        return RAGResponse(
            answer=clean_answer,
            status="insufficient_evidence",
            citations=[],
            unverified_citations=[],
            trace_id=trace_id,
            model=model,
        )

    cited_indices = extract_citation_indices(clean_answer)
    verified_citations: List[Citation] = []
    unverified_citations: List[int] = []

    for idx in cited_indices:
        # Convert 1-indexed citation to 0-indexed chunk list
        chunk_pos = idx - 1
        if 0 <= chunk_pos < len(evidence_chunks):
            chunk = evidence_chunks[chunk_pos]
            meta = chunk.metadata or {}
            source_name = meta.get("source") or meta.get("file_name") or "unknown_source"
            snippet = chunk.text[:200].replace("\n", " ")

            verified_citations.append(
                Citation(
                    citation_index=idx,
                    chunk_id=chunk.chunk_id,
                    source=source_name,
                    snippet=snippet,
                    verified=True,
                    metadata=dict(meta),
                )
            )
        else:
            unverified_citations.append(idx)

    # Determine status
    if not evidence_chunks:
        status = "insufficient_evidence"
    elif verified_citations:
        status = "answered"
    else:
        # Answered without explicit citations or with unverified citations
        status = "partial" if unverified_citations else "answered"

    return RAGResponse(
        answer=clean_answer,
        status=status,
        citations=verified_citations,
        unverified_citations=unverified_citations,
        trace_id=trace_id,
        model=model,
    )
