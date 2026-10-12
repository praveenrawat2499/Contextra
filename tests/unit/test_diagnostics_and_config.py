"""
Unit tests for Diagnostics and Configuration Module (Phase 7).
Validates health check diagnostics, readiness probes, environment parsing, and credential protection.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from src.config import AppConfig, load_config
from src.diagnostics import (
    check_api_key_configured,
    check_filesystem_health,
    check_vector_store_health,
    run_system_diagnostics,
)


class TestConfigLoading(unittest.TestCase):
    """Test environment variable loading and validation."""

    def test_default_config_types_and_values(self):
        config = load_config()
        self.assertIsInstance(config, AppConfig)
        self.assertGreater(config.retrieval_k, 0)
        self.assertGreater(config.rrf_k, 0)
        self.assertGreater(config.max_query_chars, 0)
        self.assertGreater(config.request_timeout_s, 0)
        self.assertTrue(config.embedding_model)

    def test_boolean_flag_parsing(self):
        with patch.dict(os.environ, {"ENABLE_EXPANSION": "true", "ENABLE_RERANKING": "1"}):
            config = load_config()
            self.assertTrue(config.enable_expansion)
            self.assertTrue(config.enable_reranking)


class TestDiagnostics(unittest.TestCase):
    """Test health check and readiness diagnostics."""

    def test_check_api_key_when_set(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "fake_test_key_12345"}):
            res = check_api_key_configured()
            self.assertEqual(res["status"], "configured")
            self.assertTrue(res["configured"])
            # Never expose the key value in the message
            self.assertNotIn("fake_test_key", res["message"])

    def test_check_api_key_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            res = check_api_key_configured()
            self.assertEqual(res["status"], "missing")
            self.assertFalse(res["configured"])

    def test_check_vector_store_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = check_vector_store_health(os.path.join(tmp_dir, "nonexistent"))
            self.assertEqual(res["status"], "not_found")
            self.assertFalse(res["aligned"])

    def test_check_filesystem_health_writable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = check_filesystem_health(tmp_dir)
            self.assertEqual(res["status"], "writable")
            self.assertTrue(res["writable"])

    def test_run_system_diagnostics_structure(self):
        res = run_system_diagnostics(persist_dir="faiss_store")
        self.assertIn("overall_status", res)
        self.assertIn(res["overall_status"], ("healthy", "degraded", "unready"))
        self.assertIn("components", res)
        self.assertIn("api_credentials", res["components"])
        self.assertIn("vector_store", res["components"])
        self.assertIn("filesystem", res["components"])


if __name__ == "__main__":
    unittest.main()
