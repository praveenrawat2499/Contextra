"""
Reranking Module using Cross-Encoder.
Provides configurable, bounded reranking of retrieval candidates with safe fallback.
"""

from __future__ import annotations

import copy
from typing import List, Optional

from src.logger import get_logger, trace_span
from src.retrieval_types import RetrievalResult

logger = get_logger("rag.reranker")


class Reranker:
    """Optional cross-encoder reranker for candidate passages."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        enabled: bool = False,
        max_candidates: int = 15,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self.max_candidates = max_candidates
        self._model = None
        self._model_failed = False

    def _get_model(self):
        """Lazy load cross-encoder model."""
        if self._model is None and not self._model_failed:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading cross-encoder reranker model: {self.model_name}...")
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Cross-encoder reranker '{self.model_name}' loaded successfully.")
            except Exception as exc:
                self._model_failed = True
                logger.error(
                    f"Failed to load CrossEncoder '{self.model_name}': {type(exc).__name__}: {exc}. "
                    f"Reranking will be disabled for future queries."
                )
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 5,
        trace_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Rerank a candidate pool using query-passage cross-encoder scores.

        Guarantees:
        - If disabled or empty candidates, returns input candidates[:top_k] without scoring overhead.
        - Limits candidate pool to max_candidates.
        - Preserves stable chunk IDs, original metadata, and text.
        - Deterministic tie-breaking by chunk_id when scores are equal.
        - Gracefully falls back to input rankings if model fails.
        """
        if not candidates:
            return []

        if not self.enabled:
            return candidates[:top_k]

        with trace_span("reranking", trace_id, logger=logger) as span:
            span["initial_candidates"] = len(candidates)
            pool = candidates[: self.max_candidates]

            model = self._get_model()
            if model is None:
                logger.warning(
                    f"[{trace_id or 'local'}] Reranker model unavailable. Falling back to input ranking."
                )
                span["status"] = "fallback"
                return pool[:top_k]

            try:
                pairs = [(query, c.text) for c in pool]
                raw_scores = model.predict(pairs)

                scored_candidates: List[tuple[float, str, RetrievalResult]] = []
                for score, orig in zip(raw_scores, pool):
                    score_val = float(score)
                    scored_candidates.append((score_val, orig.chunk_id, orig))

                # Deterministic sort: descending score, ascending chunk_id for tie-breaking
                scored_candidates.sort(key=lambda item: (-item[0], item[1]))

                reranked_results: List[RetrievalResult] = []
                for rank, (score, chunk_id, orig) in enumerate(scored_candidates[:top_k], start=1):
                    new_meta = copy.deepcopy(orig.metadata)
                    new_meta["rerank_score"] = round(score, 6)
                    new_meta["pre_rerank_rank"] = orig.rank
                    new_meta["pre_rerank_score"] = orig.score

                    reranked_results.append(
                        RetrievalResult(
                            chunk_id=orig.chunk_id,
                            text=orig.text,
                            metadata=new_meta,
                            retriever=f"{orig.retriever}+rerank",
                            rank=rank,
                            score=score,
                            score_type="cross_encoder",
                            provenance=copy.deepcopy(orig.provenance),
                        )
                    )

                span["reranked_count"] = len(reranked_results)
                span["top_score"] = round(reranked_results[0].score, 4) if reranked_results else None
                logger.info(
                    f"[{trace_id or 'local'}] Reranked {len(pool)} candidates -> top {len(reranked_results)} results."
                )
                return reranked_results

            except Exception as exc:
                logger.error(
                    f"[{trace_id or 'local'}] Reranking inference error ({type(exc).__name__}: {exc}). "
                    f"Falling back to original candidate ordering."
                )
                span["status"] = "inference_error"
                return pool[:top_k]
