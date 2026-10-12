"""
Unit tests for Query Expansion and Cross-Encoder Reranking (Phase 5).
Validates parser robustness, feature toggles, tie-breaking, bounding, and fallbacks.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.query_transform import (
    expand_query,
    expand_query_heuristic,
    parse_expansion_output,
)
from src.reranker import Reranker
from src.search import RetrievalResult


class TestQueryExpansionParsing(unittest.TestCase):
    """Test output parsing, bounding, and deduplication of query expansion."""

    def test_original_query_preserved_as_first_element(self):
        raw = "1. gradient descent formulas\n2. learning rate tuning"
        parsed = parse_expansion_output(raw, original_query="What is gradient descent?", max_queries=3)
        self.assertEqual(parsed[0], "What is gradient descent?")
        self.assertIn("gradient descent formulas", parsed)

    def test_strips_numbering_and_bullet_points(self):
        raw = (
            "1. backpropagation step by step\n"
            "- neural network backpropagation\n"
            "* computing gradients with chain rule\n"
            "4) error backward pass"
        )
        parsed = parse_expansion_output(raw, original_query="backprop", max_queries=4)
        self.assertEqual(parsed[0], "backprop")
        self.assertIn("backpropagation step by step", parsed)
        self.assertIn("neural network backpropagation", parsed)
        self.assertIn("computing gradients with chain rule", parsed)

    def test_deduplicates_case_insensitively(self):
        raw = (
            "1. Deep Learning\n"
            "2. deep learning\n"
            "3. DEEP LEARNING\n"
            "4. neural networks"
        )
        parsed = parse_expansion_output(raw, original_query="deep learning", max_queries=3)
        # Should contain 'deep learning' and 'neural networks' only once
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0], "deep learning")
        self.assertEqual(parsed[1], "neural networks")

    def test_ignores_conversational_filler_and_markdown(self):
        raw = (
            "Here are alternative queries for your search:\n"
            "```\n"
            "1. loss function calculation\n"
            "Sure! Another query:\n"
            "cost function optimization\n"
            "```"
        )
        parsed = parse_expansion_output(raw, original_query="loss function", max_queries=3)
        self.assertEqual(parsed[0], "loss function")
        self.assertIn("loss function calculation", parsed)
        self.assertIn("cost function optimization", parsed)

    def test_bounds_output_to_max_queries(self):
        raw = "\n".join([f"query variation {i}" for i in range(10)])
        parsed = parse_expansion_output(raw, original_query="base query", max_queries=3)
        self.assertEqual(len(parsed), 3)

    def test_bounds_query_length_chars(self):
        long_query = "word " * 100
        parsed = parse_expansion_output(long_query, original_query="base", max_queries=2, max_chars=50)
        self.assertLessEqual(len(parsed[1]), 50)

    def test_empty_or_whitespace_output_returns_only_original(self):
        parsed = parse_expansion_output("   \n\n  ", original_query="test query")
        self.assertEqual(parsed, ["test query"])

    def test_empty_original_query_returns_empty(self):
        parsed = parse_expansion_output("1. some query", original_query="")
        self.assertEqual(parsed, [])


class TestQueryExpansionBehavior(unittest.TestCase):
    """Test expand_query orchestration, flags, and fallbacks."""

    def test_disabled_expansion_returns_only_original_without_calling_llm(self):
        mock_llm = MagicMock()
        results = expand_query("what is attention", llm=mock_llm, enable_expansion=False)
        self.assertEqual(results, ["what is attention"])
        mock_llm.invoke.assert_not_called()

    def test_heuristic_expansion_strips_common_prefixes(self):
        results = expand_query_heuristic("What is backpropagation?", max_queries=2)
        self.assertEqual(results[0], "What is backpropagation?")
        self.assertEqual(results[1], "backpropagation")

    def test_llm_exception_falls_back_gracefully_to_original(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API timeout")

        results = expand_query("what is gradient descent", llm=mock_llm, enable_expansion=True)
        self.assertEqual(results, ["what is gradient descent"])


class TestRerankerUnit(unittest.TestCase):
    """Test cross-encoder reranker logic, toggles, tie-breaking, and metadata preservation."""

    def setUp(self):
        self.c1 = RetrievalResult(
            chunk_id="chunk_01",
            text="Gradient descent optimizes weights using the learning rate.",
            metadata={"source": "ml.txt", "doc_id": "d1"},
            retriever="faiss",
            rank=1,
            score=0.8,
        )
        self.c2 = RetrievalResult(
            chunk_id="chunk_02",
            text="Attention mechanism dynamically weights input tokens.",
            metadata={"source": "transformer.txt", "doc_id": "d2"},
            retriever="bm25",
            rank=2,
            score=0.5,
        )

    def test_disabled_reranker_returns_unmodified_candidates(self):
        reranker = Reranker(enabled=False)
        output = reranker.rerank("gradient descent", [self.c1, self.c2], top_k=2)
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].chunk_id, "chunk_01")
        self.assertEqual(output[1].chunk_id, "chunk_02")

    def test_empty_candidates_returns_empty(self):
        reranker = Reranker(enabled=True)
        self.assertEqual(reranker.rerank("query", []), [])

    def test_reranker_updates_ordering_based_on_cross_encoder_scores(self):
        reranker = Reranker(enabled=True)
        mock_model = MagicMock()
        # Make chunk_02 score higher than chunk_01
        mock_model.predict.return_value = [0.1, 0.9]
        reranker._model = mock_model

        output = reranker.rerank("attention tokens", [self.c1, self.c2], top_k=2)
        self.assertEqual(len(output), 2)
        # chunk_02 should now be rank 1
        self.assertEqual(output[0].chunk_id, "chunk_02")
        self.assertEqual(output[0].rank, 1)
        self.assertAlmostEqual(output[0].score, 0.9)
        self.assertEqual(output[0].metadata["rerank_score"], 0.9)
        self.assertEqual(output[0].metadata["source"], "transformer.txt")

        # chunk_01 should now be rank 2
        self.assertEqual(output[1].chunk_id, "chunk_01")
        self.assertEqual(output[1].rank, 2)
        self.assertAlmostEqual(output[1].score, 0.1)

    def test_deterministic_tie_breaking_by_chunk_id(self):
        reranker = Reranker(enabled=True)
        mock_model = MagicMock()
        # Identical scores
        mock_model.predict.return_value = [0.5, 0.5]
        reranker._model = mock_model

        c_beta = RetrievalResult(
            chunk_id="beta_chunk",
            text="Text beta",
            metadata={},
            retriever="faiss",
            rank=1,
            score=0.5,
        )
        c_alpha = RetrievalResult(
            chunk_id="alpha_chunk",
            text="Text alpha",
            metadata={},
            retriever="faiss",
            rank=2,
            score=0.5,
        )

        output = reranker.rerank("query", [c_beta, c_alpha], top_k=2)
        # alpha_chunk comes before beta_chunk lexicographically
        self.assertEqual(output[0].chunk_id, "alpha_chunk")
        self.assertEqual(output[1].chunk_id, "beta_chunk")

    def test_model_failure_falls_back_to_original_order(self):
        reranker = Reranker(enabled=True)
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CUDA out of memory")
        reranker._model = mock_model

        output = reranker.rerank("query", [self.c1, self.c2], top_k=2)
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].chunk_id, "chunk_01")
        self.assertEqual(output[1].chunk_id, "chunk_02")


if __name__ == "__main__":
    unittest.main()
