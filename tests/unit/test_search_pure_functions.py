"""
Unit tests for pure / near-pure functions in src/search.py.

Tests in this file do NOT touch the filesystem, FAISS, BM25, or any
LLM/API endpoint. They rely only on in-process Python logic.

Covered:
  - RAGSearch._normalize_text   (static method)
  - RAGSearch._format_history   (static method)
  - RAGSearch._reciprocal_rank_fusion  (instance method using self.rrf_k only)
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.search import RAGSearch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bare_rag(rrf_k: int = 60) -> RAGSearch:
    """
    Create a RAGSearch with __new__ so __init__ is never called.
    Only sets the attributes needed for _reciprocal_rank_fusion.
    """
    rag = RAGSearch.__new__(RAGSearch)
    rag.rrf_k = rrf_k
    return rag


# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_collapses_internal_whitespace(self):
        assert RAGSearch._normalize_text("hello   world") == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        assert RAGSearch._normalize_text("  hello world  ") == "hello world"

    def test_lowercases_ascii(self):
        assert RAGSearch._normalize_text("Hello WORLD") == "hello world"

    def test_empty_string(self):
        assert RAGSearch._normalize_text("") == ""

    def test_single_word(self):
        assert RAGSearch._normalize_text("Python") == "python"

    def test_newlines_treated_as_whitespace(self):
        assert RAGSearch._normalize_text("hello\n\nworld") == "hello world"

    def test_tabs_treated_as_whitespace(self):
        assert RAGSearch._normalize_text("hello\tworld") == "hello world"

    def test_unicode_letters_lowercased(self):
        result = RAGSearch._normalize_text("Über die Welt")
        assert result == result.casefold()


# ---------------------------------------------------------------------------
# _format_history
# ---------------------------------------------------------------------------


class TestFormatHistory:
    def test_none_returns_empty_string(self):
        assert RAGSearch._format_history(None) == ""

    def test_empty_list_returns_empty_string(self):
        assert RAGSearch._format_history([]) == ""

    def test_formats_user_message(self):
        history = [{"role": "user", "content": "Hello"}]
        result = RAGSearch._format_history(history)
        assert "User: Hello" in result

    def test_formats_assistant_message(self):
        history = [{"role": "assistant", "content": "Hi there"}]
        result = RAGSearch._format_history(history)
        assert "Assistant: Hi there" in result

    def test_formats_alternating_turns(self):
        history = [
            {"role": "user", "content": "What is ML?"},
            {"role": "assistant", "content": "Machine learning is..."},
        ]
        result = RAGSearch._format_history(history)
        assert "User: What is ML?" in result
        assert "Assistant: Machine learning is..." in result

    def test_unknown_role_excluded(self):
        history = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = RAGSearch._format_history(history)
        assert "You are helpful" not in result
        assert "system" not in result.lower()
        assert "User: Hello" in result

    def test_whitespace_only_content_excluded(self):
        history = [{"role": "user", "content": "   "}]
        result = RAGSearch._format_history(history)
        assert result == ""

    def test_non_string_content_excluded(self):
        history = [{"role": "user", "content": 42}]  # type: ignore[list-item]
        result = RAGSearch._format_history(history)
        assert result == ""

    def test_caps_to_8_most_recent_messages(self):
        # 12 messages → only last 8 should appear
        history = [
            {"role": "user", "content": f"msg_{i:02d}"} for i in range(12)
        ]
        result = RAGSearch._format_history(history)
        # msg_00 … msg_03 should be dropped
        for i in range(4):
            assert f"msg_{i:02d}" not in result
        # msg_04 … msg_11 should be present
        for i in range(4, 12):
            assert f"msg_{i:02d}" in result

    def test_fewer_than_8_messages_all_included(self):
        history = [{"role": "user", "content": f"msg_{i:02d}"} for i in range(5)]
        result = RAGSearch._format_history(history)
        for i in range(5):
            assert f"msg_{i:02d}" in result


# ---------------------------------------------------------------------------
# _reciprocal_rank_fusion
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_single_faiss_list_preserves_rank_order(self):
        rag = _bare_rag()
        faiss_results = [
            {"metadata": {"text": "chunk A"}},
            {"metadata": {"text": "chunk B"}},
            {"metadata": {"text": "chunk C"}},
        ]
        result = rag._reciprocal_rank_fusion(faiss_results, [], top_k=3)
        assert result == ["chunk A", "chunk B", "chunk C"]

    def test_top_k_limits_output_length(self):
        rag = _bare_rag()
        faiss_results = [{"metadata": {"text": f"chunk {i}"}} for i in range(10)]
        result = rag._reciprocal_rank_fusion(faiss_results, [], top_k=3)
        assert len(result) == 3

    def test_chunk_in_both_retrievers_ranked_higher(self):
        """
        A chunk that appears in both FAISS and BM25 results accumulates
        score from two lists and should rank above a chunk in only one.
        """
        rag = _bare_rag()
        faiss_results = [
            {"metadata": {"text": "unique faiss chunk"}},
            {"metadata": {"text": "shared chunk"}},
        ]
        bm25_docs = [
            Document(page_content="shared chunk"),
            Document(page_content="unique bm25 chunk"),
        ]
        result = rag._reciprocal_rank_fusion(faiss_results, bm25_docs, top_k=4)
        assert result[0] == "shared chunk", (
            f"Expected 'shared chunk' first (appears in both retrievers), got {result}"
        )

    def test_empty_faiss_and_bm25_returns_empty(self):
        rag = _bare_rag()
        result = rag._reciprocal_rank_fusion([], [], top_k=5)
        assert result == []

    def test_empty_bm25_returns_faiss_results(self):
        rag = _bare_rag()
        faiss_results = [{"metadata": {"text": "only faiss"}}]
        result = rag._reciprocal_rank_fusion(faiss_results, [], top_k=5)
        assert result == ["only faiss"]

    def test_empty_faiss_returns_bm25_results(self):
        rag = _bare_rag()
        bm25_docs = [Document(page_content="only bm25")]
        result = rag._reciprocal_rank_fusion([], bm25_docs, top_k=5)
        assert result == ["only bm25"]

    def test_duplicate_within_single_retriever_counted_once(self):
        """Exact duplicate text in one retriever list must not inflate score."""
        rag = _bare_rag()
        faiss_results = [
            {"metadata": {"text": "dup chunk"}},
            {"metadata": {"text": "dup chunk"}},  # exact duplicate
            {"metadata": {"text": "other chunk"}},
        ]
        result = rag._reciprocal_rank_fusion(faiss_results, [], top_k=10)
        assert result.count("dup chunk") == 1

    def test_whitespace_only_chunks_excluded(self):
        rag = _bare_rag()
        faiss_results = [
            {"metadata": {"text": "   "}},
            {"metadata": {"text": "\n\n"}},
            {"metadata": {"text": "valid chunk"}},
        ]
        result = rag._reciprocal_rank_fusion(faiss_results, [], top_k=10)
        assert "valid chunk" in result
        assert "   " not in result
        assert "\n\n" not in result

    def test_missing_metadata_key_excluded(self):
        """Results with no metadata dict should be silently skipped."""
        rag = _bare_rag()
        faiss_results = [
            {"metadata": None},
            {"metadata": {"text": "good chunk"}},
        ]
        result = rag._reciprocal_rank_fusion(faiss_results, [], top_k=5)
        assert "good chunk" in result

    def test_rrf_k_parameter_affects_scores(self):
        """
        Lower rrf_k means rank differences have larger impact.
        With rrf_k=1 the gap between rank-1 and rank-2 is larger than with rrf_k=60.
        This test verifies the ordering is still correct (rank-1 wins).
        """
        rag_low = _bare_rag(rrf_k=1)
        rag_high = _bare_rag(rrf_k=60)
        faiss_results = [
            {"metadata": {"text": "first"}},
            {"metadata": {"text": "second"}},
        ]
        assert rag_low._reciprocal_rank_fusion(faiss_results, [], top_k=2)[0] == "first"
        assert rag_high._reciprocal_rank_fusion(faiss_results, [], top_k=2)[0] == "first"
