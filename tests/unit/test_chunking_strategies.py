"""
Unit tests for Phase 4: Chunking Strategies and Configurations.

Covers:
  - Parameter validation (chunk_size > 0, chunk_overlap < chunk_size)
  - Strategy validation (recursive vs structure_aware)
  - Empty document handling
  - Semantic boundary splitting in structure_aware mode
  - Overlap preservation between adjacent chunks
  - Metadata preservation across chunking
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.embedding import EmbeddingPipeline


class TestChunkingValidation:
    def test_invalid_chunk_size_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            EmbeddingPipeline.__new__(EmbeddingPipeline).__init__(
                model_name="all-MiniLM-L6-v2", chunk_size=0, chunk_overlap=0
            )

    def test_negative_overlap_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            EmbeddingPipeline.__new__(EmbeddingPipeline).__init__(
                model_name="all-MiniLM-L6-v2", chunk_size=500, chunk_overlap=-10
            )

    def test_overlap_greater_equal_to_size_raises_value_error(self):
        with pytest.raises(ValueError, match="must be strictly less than"):
            EmbeddingPipeline.__new__(EmbeddingPipeline).__init__(
                model_name="all-MiniLM-L6-v2", chunk_size=500, chunk_overlap=500
            )

    def test_unsupported_strategy_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported strategy"):
            EmbeddingPipeline.__new__(EmbeddingPipeline).__init__(
                model_name="all-MiniLM-L6-v2", chunk_size=500, chunk_overlap=100, strategy="invalid_strategy"
            )


class TestChunkingBehavior:
    def test_empty_documents_returns_empty_list(self):
        pipeline = EmbeddingPipeline.__new__(EmbeddingPipeline)
        pipeline.chunk_size = 500
        pipeline.chunk_overlap = 50
        pipeline.strategy = "recursive"
        assert pipeline.chunk_documents([]) == []

    def test_structure_aware_splits_on_markdown_headers(self):
        pipeline = EmbeddingPipeline.__new__(EmbeddingPipeline)
        pipeline.chunk_size = 120
        pipeline.chunk_overlap = 20
        pipeline.strategy = "structure_aware"

        doc = Document(
            page_content=(
                "# Machine Learning\n\n"
                "Supervised learning trains on labeled data.\n\n"
                "## Neural Networks\n\n"
                "Neural networks consist of layers of interconnected artificial neurons."
            ),
            metadata={"source": "ml_guide.md", "doc_id": "guide_01"},
        )

        chunks = pipeline.chunk_documents([doc])
        assert len(chunks) >= 2
        # Headings should not be broken arbitrarily across tiny fragments
        texts = [c.page_content for c in chunks]
        assert any("Supervised learning" in t for t in texts)
        assert any("Neural networks consist" in t for t in texts)
        for c in chunks:
            assert c.metadata["source"] == "ml_guide.md"
            assert c.metadata["doc_id"] == "guide_01"
            assert "chunk_id" in c.metadata

    def test_overlap_preserves_context_across_chunks(self):
        pipeline = EmbeddingPipeline.__new__(EmbeddingPipeline)
        pipeline.chunk_size = 60
        pipeline.chunk_overlap = 20
        pipeline.strategy = "recursive"

        doc = Document(
            page_content="Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau",
            metadata={"source": "alphabet.txt", "doc_id": "alph1"},
        )
        chunks = pipeline.chunk_documents([doc])
        assert len(chunks) >= 2

        # Adjacent chunks should share some overlapping tokens/words
        first_chunk = chunks[0].page_content
        second_chunk = chunks[1].page_content
        first_words = set(first_chunk.split())
        second_words = set(second_chunk.split())
        shared = first_words.intersection(second_words)
        assert len(shared) > 0, f"Expected overlapping words, found none: '{first_chunk}' vs '{second_chunk}'"
