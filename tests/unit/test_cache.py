import time
from pathlib import Path
import threading

from src.cache import ExactQueryCache, get_index_version
from src.generation_types import Citation, RAGResponse


def _sample_response(text: str = "ML is AI.") -> RAGResponse:
    return RAGResponse(
        answer=text,
        status="answered",
        citations=[
            Citation(
                citation_index=1,
                source="machine_learning.txt",
                chunk_id="chunk_12345",
                snippet="Subset of AI",
            )
        ],
        metadata={"latency_ms": 650.0},
    )


def test_cache_miss_then_hit():
    cache = ExactQueryCache(max_entries=10, index_version="v1")
    key = cache.compute_key(
        query="What is machine learning?",
        llm_model="z-ai/glm-5.3-flash",
        top_k=5,
    )

    # 1. Initial miss
    assert cache.get(key) is None
    assert cache.misses == 1
    assert cache.hits == 0

    # 2. Store response
    resp = _sample_response()
    cache.set(key, resp)

    # 3. Hit
    cached_resp = cache.get(key)
    assert cached_resp is not None
    assert cached_resp.answer == "ML is AI."
    assert len(cached_resp.citations) == 1
    assert cached_resp.citations[0].chunk_id == "chunk_12345"
    assert cache.hits == 1
    assert cache.get_stats()["llm_calls_avoided"] == 1


def test_query_normalization():
    cache = ExactQueryCache(max_entries=10, index_version="v1")

    # Queries with different casing and internal whitespace
    key1 = cache.compute_key(
        query="What is Machine Learning?",
        llm_model="z-ai/glm-5.3-flash",
        top_k=5,
    )
    key2 = cache.compute_key(
        query="   what is   machine learning?  ",
        llm_model="z-ai/glm-5.3-flash",
        top_k=5,
    )

    assert key1 == key2

    cache.set(key1, _sample_response())
    hit = cache.get(key2)
    assert hit is not None
    assert hit.answer == "ML is AI."


def test_parameter_isolation():
    cache = ExactQueryCache(max_entries=10, index_version="v1")

    key_k5 = cache.compute_key("test query", "model_a", top_k=5)
    key_k10 = cache.compute_key("test query", "model_a", top_k=10)
    key_model_b = cache.compute_key("test query", "model_b", top_k=5)

    assert key_k5 != key_k10
    assert key_k5 != key_model_b

    cache.set(key_k5, _sample_response("k=5 answer"))
    assert cache.get(key_k10) is None
    assert cache.get(key_model_b) is None


def test_index_version_invalidation():
    cache = ExactQueryCache(max_entries=10, index_version="v1")
    key = cache.compute_key("query", "model", top_k=5)

    cache.set(key, _sample_response())
    assert cache.get(key) is not None

    # Update index version to simulate document rebuild
    cache.update_index_version("v2")

    # Cache should be purged
    assert cache.get(key) is None
    assert cache.get_stats()["size"] == 0


def test_deepcopy_isolation():
    cache = ExactQueryCache(max_entries=10, index_version="v1")
    key = cache.compute_key("query", "model", top_k=5)

    orig = _sample_response("Original answer")
    cache.set(key, orig)

    # Mutate retrieved object
    retrieved = cache.get(key)
    retrieved.answer = "Mutated answer"
    retrieved.citations.clear()

    # Second fetch must remain unmodified
    fresh = cache.get(key)
    assert fresh.answer == "Original answer"
    assert len(fresh.citations) == 1


def test_lru_capacity_eviction():
    cache = ExactQueryCache(max_entries=2, index_version="v1")

    k1 = cache.compute_key("q1", "m", 5)
    k2 = cache.compute_key("q2", "m", 5)
    k3 = cache.compute_key("q3", "m", 5)

    cache.set(k1, _sample_response("ans 1"))
    cache.set(k2, _sample_response("ans 2"))
    assert cache.get_stats()["size"] == 2

    # Access k1 to make k2 least recently used
    _ = cache.get(k1)

    # Add k3, which should evict k2
    cache.set(k3, _sample_response("ans 3"))

    assert cache.get(k1) is not None
    assert cache.get(k3) is not None
    assert cache.get(k2) is None
    assert cache.get_stats()["evictions"] == 1


def test_ttl_expiration():
    cache = ExactQueryCache(max_entries=10, ttl_seconds=1, index_version="v1")
    key = cache.compute_key("quick query", "m", 5)

    cache.set(key, _sample_response())
    assert cache.get(key) is not None

    time.sleep(1.1)
    # Expired entry should return None
    assert cache.get(key) is None


def test_thread_safety_concurrency():
    cache = ExactQueryCache(max_entries=100, index_version="v1")
    key = cache.compute_key("concurrent_query", "m", 5)
    cache.set(key, _sample_response())

    def worker():
        for _ in range(50):
            res = cache.get(key)
            assert res is not None

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert cache.hits == 250


def test_get_index_version_deterministic(tmp_path):
    # Empty dir
    v1 = get_index_version(tmp_path)
    assert isinstance(v1, str)
    assert len(v1) == 16

    # Create dummy index files
    (tmp_path / "metadata.pkl").write_text("hello", encoding="utf-8")
    (tmp_path / "faiss.index").write_text("world", encoding="utf-8")

    v2 = get_index_version(tmp_path)
    assert v2 != v1

    # Same state gives same version
    v3 = get_index_version(tmp_path)
    assert v3 == v2


def test_rag_search_cache_hit_avoids_llm():
    from unittest.mock import MagicMock
    from src.search import RAGSearch

    search = RAGSearch.__new__(RAGSearch)
    search.llm_model = "test-model"
    search.enable_expansion = False
    search.enable_reranking = False
    search.index_version = "v1"
    search.query_cache = ExactQueryCache(max_entries=10, index_version="v1")
    search.llm = MagicMock()
    search.llm.invoke.return_value = MagicMock(content="Machine learning is AI [Context 1].")
    search.retrieve_and_rank = MagicMock(return_value=[
        MagicMock(chunk_id="c1", text="Machine learning is AI.", metadata={"source": "doc.txt"})
    ])

    # First call: cache miss, calls LLM
    r1 = search.generate_grounded_answer("What is ML?", top_k=1)
    assert r1.status == "answered"
    assert search.llm.invoke.call_count == 1
    assert search.get_cache_stats()["hits"] == 0
    assert search.get_cache_stats()["misses"] == 1

    # Second call (with different casing/whitespace): cache hit, avoids LLM!
    r2 = search.generate_grounded_answer("  what is ml? ", top_k=1)
    assert r2.status == "answered"
    assert search.llm.invoke.call_count == 1  # LLM call was avoided!
    assert search.get_cache_stats()["hits"] == 1
    assert r2.metadata.get("cache_hit") is True


def test_semantic_cache_paraphrase_hit():
    import numpy as np
    from src.cache import SemanticQueryCache

    cache = SemanticQueryCache(similarity_threshold=0.94, index_version="v1")

    # Normalized dummy embeddings with high cosine similarity (0.97)
    v1 = np.array([1.0, 0.1, 0.0], dtype=np.float32)
    v1 /= np.linalg.norm(v1)
    v2 = np.array([0.98, 0.2, 0.0], dtype=np.float32)
    v2 /= np.linalg.norm(v2)

    resp = _sample_response("Attention connects queries to keys.")
    cache.set(
        query="What is attention mechanism?",
        query_embedding=v1,
        response=resp,
        llm_model="test-llm",
        top_k=5,
        enable_expansion=False,
        enable_reranking=False,
    )

    # Paraphrase query
    match = cache.get(
        query="Explain attention mechanism",
        query_embedding=v2,
        llm_model="test-llm",
        top_k=5,
        enable_expansion=False,
        enable_reranking=False,
    )

    assert match is not None
    cached_resp, sim = match
    assert sim >= 0.94
    assert cached_resp.answer == "Attention connects queries to keys."
    assert cache.hits == 1


def test_semantic_cache_contradiction_guard():
    import numpy as np
    from src.cache import SemanticQueryCache

    cache = SemanticQueryCache(similarity_threshold=0.90, polarity_guard=True, index_version="v1")

    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.99, 0.01, 0.0], dtype=np.float32)

    resp = _sample_response("Attention is essential.")
    cache.set(
        query="Is attention needed?",
        query_embedding=v1,
        response=resp,
        llm_model="test-llm",
        top_k=5,
        enable_expansion=False,
        enable_reranking=False,
    )

    # Contradictory query with negative polarity flip
    match = cache.get(
        query="Is attention not needed?",
        query_embedding=v2,
        llm_model="test-llm",
        top_k=5,
        enable_expansion=False,
        enable_reranking=False,
    )

    # Polarity guard MUST reject the match to prevent answering opposite claim
    assert match is None
    assert cache.misses == 1


def test_semantic_cache_unanswerable_preservation():
    import numpy as np
    from src.cache import SemanticQueryCache

    cache = SemanticQueryCache(similarity_threshold=0.94, index_version="v1")

    v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.05, 0.99, 0.0], dtype=np.float32)
    v2 /= np.linalg.norm(v2)

    unans_resp = RAGResponse(
        answer="I couldn't find relevant documents for your question.",
        status="insufficient_evidence",
        citations=[],
    )
    cache.set(
        query="What is quantum teleportation in Transformers?",
        query_embedding=v1,
        response=unans_resp,
        llm_model="test-llm",
        top_k=5,
        enable_expansion=False,
        enable_reranking=False,
    )

    match = cache.get(
        query="How does quantum teleportation work in Transformers?",
        query_embedding=v2,
        llm_model="test-llm",
        top_k=5,
        enable_expansion=False,
        enable_reranking=False,
    )

    assert match is not None
    cached_resp, _ = match
    assert cached_resp.status == "insufficient_evidence"
    assert len(cached_resp.citations) == 0


def test_semantic_cache_document_update_invalidation():
    import numpy as np
    from src.cache import SemanticQueryCache

    cache = SemanticQueryCache(similarity_threshold=0.90, index_version="v1")
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    cache.set(
        query="What is Python?",
        query_embedding=v1,
        response=_sample_response("Python is a language."),
        llm_model="test-llm",
        top_k=5,
        enable_expansion=False,
        enable_reranking=False,
    )

    # Index rebuild advances version to v2
    cache.update_index_version("v2")

    # Stale entry must be invalidated
    match = cache.get(
        query="What is Python?",
        query_embedding=v1,
        llm_model="test-llm",
        top_k=5,
        enable_expansion=False,
        enable_reranking=False,
    )
    assert match is None
    assert cache.get_stats()["size"] == 0


def test_rag_search_two_tier_flow():
    from unittest.mock import MagicMock
    import numpy as np
    from src.search import RAGSearch

    search = RAGSearch.__new__(RAGSearch)
    search.llm_model = "test-model"
    search.enable_expansion = False
    search.enable_reranking = False
    search.index_version = "v1"
    search.query_cache = ExactQueryCache(max_entries=10, index_version="v1")

    from src.cache import SemanticQueryCache
    search.semantic_cache = SemanticQueryCache(similarity_threshold=0.94, index_version="v1")

    # Mock encoder
    v_base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_para = np.array([0.99, 0.05, 0.0], dtype=np.float32)
    v_para /= np.linalg.norm(v_para)

    def mock_encode(text, normalize_embeddings=True):
        if "explain" in text.lower():
            return v_para
        return v_base

    search.vectorstore = MagicMock()
    search.vectorstore.model.encode = mock_encode

    search.llm = MagicMock()
    search.llm.invoke.return_value = MagicMock(content="Attention is all you need [Context 1].")
    search.retrieve_and_rank = MagicMock(return_value=[
        MagicMock(chunk_id="c1", text="Attention is all you need.", metadata={"source": "paper.pdf"})
    ])

    # 1. Cold query -> calls LLM once, populates exact and semantic cache
    r1 = search.generate_grounded_answer("What is attention?", top_k=1)
    assert search.llm.invoke.call_count == 1
    assert r1.status == "answered"

    # 2. Exact match query -> served from Tier 1 (exact)
    r2 = search.generate_grounded_answer("what is attention?", top_k=1)
    assert search.llm.invoke.call_count == 1  # No LLM call!
    assert r2.metadata.get("cache_tier") == "exact"

    # 3. Paraphrase query -> misses Tier 1, served from Tier 2 (semantic)
    r3 = search.generate_grounded_answer("Can you explain attention?", top_k=1)
    assert search.llm.invoke.call_count == 1  # Still NO new LLM call!
    assert r3.metadata.get("cache_tier") == "semantic"
    assert r3.metadata.get("similarity") >= 0.94


