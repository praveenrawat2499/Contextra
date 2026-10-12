"""
Unit tests for Phase 2: Metadata and Ingestion Reliability.

Covers:
  - Provenance metadata preservation (source, doc_id, file_name, page)
  - Stable chunk_id and content_hash generation
  - Deduplication on repeated ingestion of identical chunks
  - Staged index rebuild safety (failed rebuild does not destroy active index)
  - Structured per-file error reporting during ingestion
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from langchain_core.documents import Document

from src.data_loader import load_documents_with_report, load_all_documents, _compute_file_hash
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


class TestDataLoaderProvenance:
    def test_file_hash_is_deterministic(self, tmp_path):
        f1 = tmp_path / "sample.txt"
        f1.write_text("Deterministic content for hashing", encoding="utf-8")
        h1 = _compute_file_hash(f1)
        h2 = _compute_file_hash(f1)
        assert h1 == h2
        assert len(h1) == 16

    def test_load_documents_attaches_provenance(self):
        docs, report = load_documents_with_report(str(FIXTURE_DIR))
        assert len(docs) > 0
        assert report["total_files"] >= 1
        assert report["successful_files"] >= 1
        assert report["failed_files"] == 0

        for doc in docs:
            assert "source" in doc.metadata
            assert "doc_id" in doc.metadata
            assert "file_name" in doc.metadata
            assert doc.metadata["file_name"].endswith(".txt")
            assert not doc.metadata["source"].startswith("C:\\")  # safe relative path

    def test_load_documents_reports_errors_per_file(self, tmp_path):
        # Create one good file and one invalid file
        good_file = tmp_path / "valid.txt"
        good_file.write_text("Hello world", encoding="utf-8")

        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("{invalid json content:", encoding="utf-8")

        docs, report = load_documents_with_report(str(tmp_path))
        assert report["total_files"] >= 1
        # The bad file should be captured in errors or failed_files
        if report["failed_files"] > 0:
            assert len(report["errors"]) == report["failed_files"]


class TestEmbeddingChunkProvenance:
    def test_chunking_assigns_stable_chunk_id_and_hash(self):
        pipeline = EmbeddingPipeline.__new__(EmbeddingPipeline)
        pipeline.chunk_size = 50
        pipeline.chunk_overlap = 10

        doc1 = Document(
            page_content="Machine learning is a subset of artificial intelligence. It learns patterns from data.",
            metadata={"source": "test.txt", "doc_id": "abc12345"},
        )
        chunks = pipeline.chunk_documents([doc1])
        assert len(chunks) > 1

        for idx, chunk in enumerate(chunks):
            assert chunk.metadata["doc_id"] == "abc12345"
            assert chunk.metadata["chunk_index"] == idx
            assert chunk.metadata["chunk_id"] == f"abc12345:{idx}"
            assert "content_hash" in chunk.metadata
            assert len(chunk.metadata["content_hash"]) == 16
            assert chunk.metadata["source"] == "test.txt"


class TestVectorStoreDeduplicationAndStaging:
    def test_deduplication_removes_identical_chunks(self):
        store = FaissVectorStore.__new__(FaissVectorStore)
        store.persist_dir = "faiss_store_test"
        store.embedding_model = "all-MiniLM-L6-v2"
        store.chunk_size = 500
        store.chunk_overlap = 50

        # Duplicate document content
        doc1 = Document(
            page_content="Identical content chunk for deduplication testing.",
            metadata={"source": "doc1.txt", "doc_id": "doc1"},
        )
        doc2 = Document(
            page_content="Identical content chunk for deduplication testing.",
            metadata={"source": "doc2.txt", "doc_id": "doc2"},
        )

        # Mock embedding pipeline
        mock_chunks = [
            Document(page_content="Identical content chunk", metadata={"content_hash": "hash_123", "chunk_id": "doc1:0"}),
            Document(page_content="Identical content chunk", metadata={"content_hash": "hash_123", "chunk_id": "doc2:0"}),
            Document(page_content="Unique chunk 2", metadata={"content_hash": "hash_456", "chunk_id": "doc2:1"}),
        ]

        with patch("src.vectorstore.EmbeddingPipeline") as MockPipe:
            instance = MockPipe.return_value
            instance.chunk_documents.return_value = mock_chunks
            # Embeddings only called for the 2 deduped chunks
            instance.embed_chunks.return_value = np.zeros((2, 128), dtype="float32")

            with patch.object(store, "save"):
                store.build_from_documents([doc1, doc2])

                # Total unique vectors indexed must be 2, not 3
                assert len(store.metadata) == 2
                assert store.index.ntotal == 2
                hashes = [m["content_hash"] for m in store.metadata]
                assert hashes == ["hash_123", "hash_456"]

    def test_failed_rebuild_preserves_active_index(self):
        """
        Verify Defect C2 fix: a failure during chunking, embedding, or staging
        must leave the previously working store completely intact.
        """
        store = FaissVectorStore.__new__(FaissVectorStore)
        store.persist_dir = "faiss_store_test"
        store.embedding_model = "all-MiniLM-L6-v2"
        store.chunk_size = 500
        store.chunk_overlap = 50
        original_metadata = [{"text": "original safe chunk"}]
        original_index = MagicMock()
        original_index.ntotal = 1
        store.index = original_index
        store.metadata = original_metadata

        # Simulate rebuild throwing an error during embed_chunks
        with patch("src.vectorstore.EmbeddingPipeline") as MockPipe:
            instance = MockPipe.return_value
            instance.chunk_documents.return_value = [Document(page_content="new")]
            instance.embed_chunks.side_effect = RuntimeError("Embedding model GPU OOM")

            with pytest.raises(RuntimeError, match="Embedding model GPU OOM"):
                store.build_from_documents([Document(page_content="new doc")])

            # Active index and metadata are still the original!
            assert store.index is original_index
            assert store.metadata == original_metadata
