"""
Shared pytest fixtures for Contextra test suite.

All fixtures are designed to run without live API calls, network access,
or credentials. Integration tests that build a real FAISS index will
download `all-MiniLM-L6-v2` from HuggingFace Hub on first run (~90 MB,
cached after that). Unit tests use only mocks and in-memory structures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_dir() -> Path:
    """Absolute path to the test fixture document directory."""
    return FIXTURE_DIR


@pytest.fixture
def fixture_docs():
    """
    Return a list of LangChain Documents built from the test fixture files.
    These are tiny, deterministic documents with no sensitive content.
    """
    from langchain_core.documents import Document

    docs = []
    for txt_file in sorted(FIXTURE_DIR.glob("*.txt")):
        docs.append(
            Document(
                page_content=txt_file.read_text(encoding="utf-8"),
                metadata={"source": txt_file.name},
            )
        )
    return docs
