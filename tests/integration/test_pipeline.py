"""
Integration tests for the FAISS persistence round-trip.

Requires:
  - SentenceTransformer `all-MiniLM-L6-v2` (downloaded once, then cached).
  - NO API keys, NO LLM, NO network calls after model cache is warm.

The tests build a small FAISS index from the test fixture documents,
persist it to a temp directory, reload it, and verify query behaviour.

Run with:
    pytest tests/integration/ -v

To skip if the model is not cached (e.g., in CI without internet):
    pytest tests/integration/ -v -m "not slow"
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.vectorstore import FaissVectorStore


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Shared built store (built once per test session for speed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_store(tmp_path_factory):
    """
    Build a real FAISS index from fixture documents, save it, and return
    both the store instance and the tmp directory for reload tests.
    """
    tmp_dir = tmp_path_factory.mktemp("faiss_integration")

    docs = []
    for txt_file in sorted(FIXTURE_DIR.glob("*.txt")):
        docs.append(
            Document(
                page_content=txt_file.read_text(encoding="utf-8"),
                metadata={"source": txt_file.name},
            )
        )

    assert docs, "No fixture documents found — check tests/fixtures/"

    store = FaissVectorStore(
        persist_dir=str(tmp_dir),
        embedding_model=EMBEDDING_MODEL,
        chunk_size=200,
        chunk_overlap=20,
    )
    store.build_from_documents(docs)
    return store, tmp_dir


# ---------------------------------------------------------------------------
# Index integrity
# ---------------------------------------------------------------------------


class TestIndexIntegrity:
    def test_index_and_metadata_same_length(self, built_store):
        """Every vector in the FAISS index must have a corresponding metadata entry."""
        store, _ = built_store
        assert store.index.ntotal == len(store.metadata), (
            f"Vector count {store.index.ntotal} != "
            f"metadata count {len(store.metadata)}"
        )

    def test_all_metadata_has_text_key(self, built_store):
        """
        Current schema stores only {'text': ...}.
        Phase 2 will add source, chunk_id, etc.
        """
        store, _ = built_store
        for i, m in enumerate(store.metadata):
            assert "text" in m, f"Metadata entry {i} missing 'text' key: {m!r}"
            assert isinstance(m["text"], str), (
                f"Metadata entry {i} 'text' is not a string: {type(m['text'])}"
            )

    def test_metadata_text_is_non_empty(self, built_store):
        store, _ = built_store
        empty = [i for i, m in enumerate(store.metadata) if not m.get("text", "").strip()]
        assert empty == [], (
            f"Chunks with empty text at indices: {empty}"
        )


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class TestPersistenceRoundTrip:
    def test_save_creates_expected_files(self, built_store):
        _, tmp_dir = built_store
        assert (tmp_dir / "faiss.index").exists()
        assert (tmp_dir / "metadata.pkl").exists()

    def test_reload_preserves_vector_count(self, built_store):
        original, tmp_dir = built_store

        reloaded = FaissVectorStore(
            persist_dir=str(tmp_dir),
            embedding_model=EMBEDDING_MODEL,
        )
        reloaded.load()

        assert reloaded.index.ntotal == original.index.ntotal

    def test_reload_preserves_metadata_count(self, built_store):
        original, tmp_dir = built_store

        reloaded = FaissVectorStore(
            persist_dir=str(tmp_dir),
            embedding_model=EMBEDDING_MODEL,
        )
        reloaded.load()

        assert len(reloaded.metadata) == len(original.metadata)

    def test_reload_query_returns_same_top_result(self, built_store):
        original, tmp_dir = built_store

        reloaded = FaissVectorStore(
            persist_dir=str(tmp_dir),
            embedding_model=EMBEDDING_MODEL,
        )
        reloaded.load()

        query = "gradient descent learning rate"
        r1 = original.query(query, top_k=1)
        r2 = reloaded.query(query, top_k=1)

        assert r1[0]["index"] == r2[0]["index"], (
            f"Top result index differs after reload: {r1[0]['index']} vs {r2[0]['index']}"
        )


# ---------------------------------------------------------------------------
# Query behaviour
# ---------------------------------------------------------------------------


class TestQueryBehaviour:
    def test_query_returns_results(self, built_store):
        store, _ = built_store
        results = store.query("attention mechanism", top_k=2)
        assert len(results) > 0

    def test_query_results_have_required_keys(self, built_store):
        store, _ = built_store
        results = store.query("gradient", top_k=3)
        for r in results:
            assert "index" in r
            assert "distance" in r
            assert "metadata" in r

    def test_top_k_limits_result_count(self, built_store):
        store, _ = built_store
        for k in (1, 2, 3):
            results = store.query("machine learning", top_k=k)
            assert len(results) <= k, (
                f"Expected <= {k} results, got {len(results)}"
            )

    def test_relevant_chunk_ranked_first_for_specific_query(self, built_store):
        """
        Query about gradient descent should return a chunk mentioning
        'gradient' near the top. Soft check: top-3 should contain the word.
        """
        store, _ = built_store
        results = store.query("how does gradient descent minimize loss", top_k=3)
        top_texts = [
            (r["metadata"] or {}).get("text", "").lower()
            for r in results
        ]
        assert any("gradient" in t for t in top_texts), (
            f"Expected 'gradient' in top-3 chunks, got: {top_texts}"
        )

    def test_query_with_fewer_vectors_than_top_k(self, built_store):
        """
        Requesting more results than indexed vectors should not crash.
        This test verifies no exception is raised (regression protection).
        """
        store, _ = built_store
        total_vectors = store.index.ntotal
        results = store.query("anything", top_k=total_vectors + 100)
        # Should return as many results as the index holds, without error
        assert len(results) <= total_vectors + 100

    def test_provenance_metadata_preserved_in_retrieved_results(self, built_store):
        """
        Phase 2 acceptance check: source provenance metadata survives into retrieved chunks.
        """
        store, _ = built_store
        results = store.query("gradient descent", top_k=2)
        assert len(results) > 0
        for r in results:
            meta = r["metadata"]
            assert meta is not None, "Metadata should not be None for valid result"
            assert "source" in meta, f"Missing 'source' in retrieved metadata: {meta}"
            assert "chunk_id" in meta, f"Missing 'chunk_id' in retrieved metadata: {meta}"
            assert "content_hash" in meta, f"Missing 'content_hash' in retrieved metadata: {meta}"
            assert meta["source"].endswith(".txt"), f"Source should preserve filename: {meta['source']}"
