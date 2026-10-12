"""
Unit tests for Phase 3: Hybrid Retrieval and Reciprocal Rank Fusion (RRF),
along with structured logging and stage tracing.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pytest

from src.search import RetrievalResult, reciprocal_rank_fusion
from src.logger import get_logger, generate_trace_id, trace_span, LOG_FILE


class TestRetrievalResultSchema:
    def test_instantiation_with_required_fields(self):
        res = RetrievalResult(
            chunk_id="doc1:0",
            text="Machine learning chunk",
            metadata={"source": "doc1.txt", "page": 1},
            retriever="faiss",
            rank=1,
            score=0.15,
            score_type="l2_distance",
        )
        assert res.chunk_id == "doc1:0"
        assert res.text == "Machine learning chunk"
        assert res.rank == 1
        assert res.retriever == "faiss"
        assert res.score == 0.15
        assert res.metadata["source"] == "doc1.txt"


class TestPureReciprocalRankFusion:
    def test_single_list_ranks_by_input_order(self):
        faiss_list = [
            RetrievalResult("c1", "Text 1", {}, "faiss", 1),
            RetrievalResult("c2", "Text 2", {}, "faiss", 2),
            RetrievalResult("c3", "Text 3", {}, "faiss", 3),
        ]
        fused = reciprocal_rank_fusion([faiss_list], rrf_k=60, top_k=3)
        assert len(fused) == 3
        assert [item["chunk_id"] for item in fused] == ["c1", "c2", "c3"]
        assert fused[0]["rrf_score"] > fused[1]["rrf_score"]

    def test_item_in_both_lists_scores_higher(self):
        # c2 is rank 2 in faiss, but rank 1 in bm25 -> should accumulate higher score than c1 (only faiss rank 1)
        faiss_list = [
            RetrievalResult("c1", "Only in FAISS", {}, "faiss", 1),
            RetrievalResult("c2", "Shared in both", {}, "faiss", 2),
        ]
        bm25_list = [
            RetrievalResult("c2", "Shared in both", {}, "bm25", 1),
            RetrievalResult("c3", "Only in BM25", {}, "bm25", 2),
        ]
        fused = reciprocal_rank_fusion([faiss_list, bm25_list], rrf_k=60, top_k=3)
        assert fused[0]["chunk_id"] == "c2"
        expected_c2_score = (1.0 / (60 + 2)) + (1.0 / (60 + 1))
        assert pytest.approx(fused[0]["rrf_score"], abs=1e-5) == expected_c2_score

    def test_deterministic_tie_breaking_by_chunk_id(self):
        # Two items with identical score: tie broken deterministically by chunk_id string
        l1 = [RetrievalResult("beta_chunk", "B text", {}, "faiss", 1)]
        l2 = [RetrievalResult("alpha_chunk", "A text", {}, "bm25", 1)]
        fused = reciprocal_rank_fusion([[l1[0]], [l2[0]]], rrf_k=60, top_k=2)
        assert len(fused) == 2
        assert fused[0]["rrf_score"] == fused[1]["rrf_score"]
        # 'alpha_chunk' precedes 'beta_chunk' alphabetically
        assert fused[0]["chunk_id"] == "alpha_chunk"
        assert fused[1]["chunk_id"] == "beta_chunk"

    def test_empty_lists_return_empty(self):
        fused = reciprocal_rank_fusion([], rrf_k=60, top_k=5)
        assert fused == []
        fused_empty_sublists = reciprocal_rank_fusion([[], []], rrf_k=60, top_k=5)
        assert fused_empty_sublists == []

    def test_metadata_and_text_preserved_in_fused_result(self):
        res = RetrievalResult(
            chunk_id="chunk_99",
            text="Special text",
            metadata={"source": "book.pdf", "page": 42},
            retriever="faiss",
            rank=1,
        )
        fused = reciprocal_rank_fusion([[res]], rrf_k=60, top_k=1)
        assert len(fused) == 1
        assert fused[0]["metadata"] == {"source": "book.pdf", "page": 42}
        assert fused[0]["text"] == "Special text"


class TestLoggingAndTracing:
    def test_logger_writes_to_file(self):
        logger = get_logger("test_module")
        test_msg = f"Testing file log write {generate_trace_id()}"
        logger.info(test_msg)

        assert LOG_FILE.exists()
        log_content = LOG_FILE.read_text(encoding="utf-8")
        assert test_msg in log_content

    def test_trace_span_measures_latency_and_records_attributes(self):
        logger = get_logger("test_trace")
        tid = generate_trace_id()
        with trace_span("dummy_stage", trace_id=tid, logger=logger) as span:
            span["count"] = 42

        assert "duration_ms" in span
        assert span["duration_ms"] >= 0.0
        assert span["trace_id"] == tid
        assert span["stage"] == "dummy_stage"
