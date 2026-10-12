"""
Query Transformation and Expansion Module.
Provides bounded, optional query expansion with safe fallbacks and output validation.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from src.logger import get_logger, trace_span

logger = get_logger("rag.query_transform")


def parse_expansion_output(
    raw_output: str,
    original_query: str,
    max_queries: int = 3,
    max_chars: int = 200,
) -> List[str]:
    """Parse, clean, deduplicate, and bound LLM query expansion output.

    Guarantees:
    - Original query is always preserved as the first element.
    - Total queries returned <= max_queries (minimum 1).
    - Queries are capped at max_chars in length.
    - Numbering, bullet points, markdown formatting, and filler lines are stripped.
    - Case-insensitive deduplication while preserving order.
    """
    clean_original = " ".join(original_query.split()).strip()
    if not clean_original:
        return []

    collected: List[str] = [clean_original]
    seen_normalized = {clean_original.casefold()}

    if not raw_output or not raw_output.strip():
        return collected

    # Split lines and clean formatting
    lines = raw_output.strip().splitlines()
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue

        # Strip numbering, bullets, quotes: e.g. "1. query", "- query", "* query"
        cleaned = re.sub(r"^(?:\d+[\.\)]|\*|\-|\•)\s*", "", cleaned).strip()
        cleaned = cleaned.strip("\"'`;")

        # Skip conversational filler, headings, or json delimiters
        lower_line = cleaned.casefold()
        if (
            lower_line.startswith(("here are", "sure", "alternative", "queries:", "query 1", "{", "}", "```"))
            or lower_line.endswith(":")
        ):
            continue

        # Enforce character bound
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].strip()

        # Normalize whitespace
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            continue

        norm_key = cleaned.casefold()
        if norm_key not in seen_normalized:
            seen_normalized.add(norm_key)
            collected.append(cleaned)

        if len(collected) >= max_queries:
            break

    return collected


def expand_query_heuristic(query: str, max_queries: int = 3) -> List[str]:
    """Deterministic, lightweight fallback for query expansion without an LLM."""
    clean = " ".join(query.split()).strip()
    if not clean:
        return []

    variants = [clean]
    seen = {clean.casefold()}

    # Strip conversational prefixes
    prefix_patterns = [
        r"^(?:what is|what are|explain|describe|tell me about|how does|how do)\s+",
        r"^(?:can you explain|please explain|what do you know about)\s+",
    ]
    core_query = clean
    for pat in prefix_patterns:
        match = re.search(pat, clean, re.IGNORECASE)
        if match:
            core_query = clean[match.end():].strip().rstrip("?.")
            break

    if core_query and core_query.casefold() not in seen:
        seen.add(core_query.casefold())
        variants.append(core_query)

    return variants[:max_queries]


def expand_query(
    query: str,
    llm: Optional[Any] = None,
    enable_expansion: bool = False,
    max_queries: int = 3,
    trace_id: Optional[str] = None,
) -> List[str]:
    """Generate bounded search query variations with safe fallback to original query."""
    clean_query = (query or "").strip()
    if not clean_query:
        return []

    # If expansion is disabled, return original immediately
    if not enable_expansion:
        return [clean_query]

    if max_queries <= 1:
        return [clean_query]

    with trace_span("query_expansion", trace_id, logger=logger) as span:
        # If no LLM is provided, use heuristic expansion
        if llm is None:
            variants = expand_query_heuristic(clean_query, max_queries=max_queries)
            span["strategy"] = "heuristic"
            span["variant_count"] = len(variants)
            return variants

        prompt = (
            f"Generate {max_queries - 1} concise alternative search queries for retrieving technical documentation "
            f"to answer the question: '{clean_query}'.\n"
            f"Rules:\n"
            f"- Output ONLY the alternative queries, one per line.\n"
            f"- Do not include numbering, bullet points, or conversational text.\n"
            f"- Keep technical terminology exact."
        )

        try:
            response = llm.invoke(prompt)
            raw_text = getattr(response, "content", str(response))
            variants = parse_expansion_output(
                raw_output=raw_text,
                original_query=clean_query,
                max_queries=max_queries,
            )
            span["strategy"] = "llm"
            span["variant_count"] = len(variants)
            logger.info(f"[{trace_id or 'local'}] Expanded query into {len(variants)} variants: {variants}")
            return variants

        except Exception as exc:
            logger.warning(
                f"[{trace_id or 'local'}] Query expansion failed ({type(exc).__name__}: {exc}). "
                f"Falling back to original query."
            )
            span["strategy"] = "fallback_original"
            span["variant_count"] = 1
            return [clean_query]
