"""
Unit tests for FaissVectorStore in src/vectorstore.py.

These tests use mocked FAISS indices so no actual SentenceTransformer
model is loaded and no filesystem writes occur in unit tests.

Covers:
  - Correct metadata lookup for valid indices (happy path)
  - BUG C4: FAISS -1 sentinel silently wraps to last metadata element
  - Metadata out-of-bounds safety (idx >= len(metadata))
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.vectorstore import FaissVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store_with_mocked_index(
    metadata: list,
    search_indices: list[int],
    search_distances: list[float],
) -> FaissVectorStore:
    """
    Build a FaissVectorStore using __new__ so __init__ is never called
    (avoids SentenceTransformer download).  Injects a mocked FAISS index.
    """
    store = FaissVectorStore.__new__(FaissVectorStore)
    store.metadata = metadata

    mock_index = MagicMock()
    mock_index.ntotal = len(metadata)
    mock_index.search.return_value = (
        np.array([search_distances], dtype="float32"),
        np.array([search_indices], dtype="int64"),
    )
    store.index = mock_index
    return store


DUMMY_QUERY = np.zeros((1, 768), dtype="float32")


# ---------------------------------------------------------------------------
# Happy-path: valid index returns correct metadata
# ---------------------------------------------------------------------------


class TestSearchHappyPath:
    def test_single_result_returns_correct_metadata(self):
        store = _store_with_mocked_index(
            metadata=[{"text": "alpha chunk"}, {"text": "beta chunk"}],
            search_indices=[0],
            search_distances=[0.1],
        )
        results = store.search(DUMMY_QUERY, top_k=1)
        assert len(results) == 1
        assert results[0]["metadata"] == {"text": "alpha chunk"}
        assert results[0]["index"] == 0

    def test_multiple_results_returned_in_index_order(self):
        metadata = [{"text": f"chunk {i}"} for i in range(5)]
        store = _store_with_mocked_index(
            metadata=metadata,
            search_indices=[2, 0, 4],
            search_distances=[0.1, 0.5, 0.9],
        )
        results = store.search(DUMMY_QUERY, top_k=3)
        assert len(results) == 3
        assert results[0]["metadata"]["text"] == "chunk 2"
        assert results[1]["metadata"]["text"] == "chunk 0"
        assert results[2]["metadata"]["text"] == "chunk 4"

    def test_result_includes_distance_field(self):
        store = _store_with_mocked_index(
            metadata=[{"text": "a"}],
            search_indices=[0],
            search_distances=[0.42],
        )
        results = store.search(DUMMY_QUERY, top_k=1)
        assert pytest.approx(results[0]["distance"], abs=1e-5) == 0.42


# ---------------------------------------------------------------------------
# BUG C4: FAISS -1 sentinel
# ---------------------------------------------------------------------------


class TestFaissSentinelBug:
    """
    FAISS returns idx = -1 when fewer vectors exist than top_k.
    In Phase 1, idx = -1 is guarded and returns metadata = None.
    """

    def test_minus_one_sentinel_does_not_wrap_to_last_element(self):
        """
        Confirms Bug C4 is fixed: idx=-1 must NOT return metadata[-1].
        It must safely return metadata=None.
        """
        store = _store_with_mocked_index(
            metadata=[{"text": "only chunk"}],
            search_indices=[-1],       # FAISS sentinel: no second result
            search_distances=[9999.0],
        )
        results = store.search(DUMMY_QUERY, top_k=1)
        assert len(results) == 1
        assert results[0]["index"] == -1
        assert results[0]["metadata"] is None

    def test_minus_one_sentinel_should_return_none_metadata(self):
        """
        Regression test: multiple sentinel results should all have metadata=None.
        """
        store = _store_with_mocked_index(
            metadata=[{"text": "only chunk"}],
            search_indices=[-1, -1],
            search_distances=[9999.0, 9999.0],
        )
        results = store.search(DUMMY_QUERY, top_k=2)
        sentinel_results = [r for r in results if r["index"] == -1]
        assert len(sentinel_results) == 2
        for r in sentinel_results:
            assert r["metadata"] is None


# ---------------------------------------------------------------------------
# Out-of-bounds metadata (index >= len(metadata))
# ---------------------------------------------------------------------------


class TestMetadataBoundsCheck:
    def test_index_within_bounds_returns_metadata(self):
        """idx < len(metadata) → returns metadata[idx]."""
        store = _store_with_mocked_index(
            metadata=[{"text": "chunk 0"}, {"text": "chunk 1"}],
            search_indices=[1],
            search_distances=[0.2],
        )
        results = store.search(DUMMY_QUERY, top_k=1)
        assert results[0]["metadata"] == {"text": "chunk 1"}

    def test_index_out_of_bounds_returns_none(self):
        """idx >= len(metadata) → returns None."""
        store = _store_with_mocked_index(
            metadata=[{"text": "only chunk"}],  # len = 1
            search_indices=[5],                 # out of bounds
            search_distances=[0.5],
        )
        results = store.search(DUMMY_QUERY, top_k=1)
        assert results[0]["metadata"] is None


# ---------------------------------------------------------------------------
# Edge Cases & Validation (Phase 1 additions)
# ---------------------------------------------------------------------------


class TestEdgeCasesAndValidation:
    def test_empty_or_uninitialized_index_returns_empty_list(self):
        store = FaissVectorStore.__new__(FaissVectorStore)
        store.index = None
        store.metadata = []
        assert store.search(DUMMY_QUERY, top_k=5) == []

        # Empty index (ntotal == 0)
        mock_empty = MagicMock()
        mock_empty.ntotal = 0
        store.index = mock_empty
        assert store.search(DUMMY_QUERY, top_k=5) == []

    def test_zero_or_negative_top_k_returns_empty_list(self):
        store = _store_with_mocked_index(
            metadata=[{"text": "chunk"}],
            search_indices=[0],
            search_distances=[0.1],
        )
        assert store.search(DUMMY_QUERY, top_k=0) == []
        assert store.search(DUMMY_QUERY, top_k=-2) == []

    def test_alignment_validation_raises_when_mismatched(self):
        store = FaissVectorStore.__new__(FaissVectorStore)
        mock_idx = MagicMock()
        mock_idx.ntotal = 5
        store.index = mock_idx
        store.metadata = [{"text": "one"}, {"text": "two"}]  # len = 2 != 5

        with pytest.raises(ValueError, match="Index/metadata mismatch"):
            store.validate_alignment()

    def test_alignment_validation_passes_when_matched(self):
        store = FaissVectorStore.__new__(FaissVectorStore)
        mock_idx = MagicMock()
        mock_idx.ntotal = 2
        store.index = mock_idx
        store.metadata = [{"text": "one"}, {"text": "two"}]
        # Should not raise
        store.validate_alignment()

    def test_save_uninitialized_index_raises_value_error(self, tmp_path):
        store = FaissVectorStore.__new__(FaissVectorStore)
        store.persist_dir = str(tmp_path)
        store.index = None
        store.metadata = []
        with pytest.raises(ValueError, match="Cannot save uninitialized index"):
            store.save()

    def test_load_missing_files_raises_file_not_found(self, tmp_path):
        store = FaissVectorStore.__new__(FaissVectorStore)
        store.persist_dir = str(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            store.load()
