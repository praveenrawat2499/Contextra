from collections import OrderedDict
import copy
import hashlib
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional

from src.generation_types import RAGResponse


def get_index_version(persist_dir: str | Path = "faiss_store") -> str:
    """
    Compute a deterministic version token representing the state of the active index.
    Combines file modification times and sizes of metadata.pkl and faiss.index.
    """
    pdir = Path(persist_dir)
    meta_file = pdir / "metadata.pkl"
    faiss_file = pdir / "faiss.index"

    hasher = hashlib.sha256()

    for fpath in (meta_file, faiss_file):
        if fpath.exists():
            stat = fpath.stat()
            token = f"{fpath.name}:{stat.st_mtime_ns}:{stat.st_size}"
            hasher.update(token.encode("utf-8"))
        else:
            hasher.update(f"{fpath.name}:missing".encode("utf-8"))

    return hasher.hexdigest()[:16]


class ExactQueryCache:
    """
    Thread-safe, bounded, version-aware LRU exact match query cache.
    Stores pre-generated RAGResponse objects to avoid redundant retrieval and LLM calls.
    """

    def __init__(
        self,
        max_entries: int = 500,
        ttl_seconds: Optional[int] = 3600,
        index_version: Optional[str] = None,
    ):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.index_version = index_version or "default_v1"
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def compute_key(
        self,
        query: str,
        llm_model: str,
        top_k: int,
        enable_expansion: bool = False,
        enable_reranking: bool = False,
    ) -> str:
        """
        Generate a unique cryptographic hash key based on normalized query parameters
        and current index version.
        """
        normalized_q = " ".join(query.strip().lower().split())
        raw_key = (
            f"q:{normalized_q}|m:{llm_model}|k:{top_k}|"
            f"exp:{int(enable_expansion)}|rrk:{int(enable_reranking)}|"
            f"ver:{self.index_version}"
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[RAGResponse]:
        """
        Retrieve a cached response by key.
        Returns None on miss or expiration. Moves hit entry to end for LRU ordering.
        """
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            entry = self._cache[key]

            # Check index version compatibility
            if entry.get("index_version") != self.index_version:
                del self._cache[key]
                self.misses += 1
                return None

            # Check TTL
            if self.ttl_seconds and (time.time() - entry["timestamp"] > self.ttl_seconds):
                del self._cache[key]
                self.misses += 1
                return None

            # LRU update: move to end
            self._cache.move_to_end(key)
            self.hits += 1

            # Return a deep copy so caller cannot mutate cached object
            return copy.deepcopy(entry["response"])

    def set(self, key: str, response: RAGResponse) -> None:
        """
        Store a RAGResponse in the cache.
        Evicts oldest entry if capacity is exceeded.
        """
        with self._lock:
            now = time.time()

            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_entries:
                    self._cache.popitem(last=False)
                    self.evictions += 1

            self._cache[key] = {
                "response": copy.deepcopy(response),
                "timestamp": now,
                "index_version": self.index_version,
            }

    def update_index_version(self, new_version: str) -> None:
        """
        Update the index version token and clear outdated cache entries.
        """
        with self._lock:
            if new_version != self.index_version:
                self.index_version = new_version
                self._cache.clear()

    def clear(self) -> None:
        """Purge all entries from the cache."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return cache health and effectiveness metrics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 4),
                "hit_rate_pct": round(hit_rate * 100.0, 2),
                "llm_calls_avoided": self.hits,
                "evictions": self.evictions,
                "index_version": self.index_version,
            }


NEGATION_TOKENS = {
    "not", "never", "no", "neither", "nor", "cannot", "can't", "don't", "didn't",
    "won't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
    "without"
}


def extract_negation_polarity(query: str) -> bool:
    """Return True if query contains negation polarity tokens, False otherwise."""
    tokens = set("".join(c if c.isalnum() else " " for c in query.lower()).split())
    return bool(tokens & NEGATION_TOKENS)


class SemanticQueryCache:
    """
    Tier-2 Semantic Vector Cache.
    Matches queries by embedding cosine similarity (threshold >= similarity_threshold)
    to serve pre-generated answers for paraphrased and semantically equivalent questions.
    
    Guarantees:
      - Polarity contradiction guard: prevents positive/negative contradiction matches.
      - Index version invalidation: rejects entries when document store is updated.
      - Parameter matching: requires same LLM model, top_k, and search parameters.
      - Deepcopy isolation: returned responses cannot mutate cache state.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.94,
        max_entries: int = 500,
        ttl_seconds: Optional[int] = 3600,
        index_version: Optional[str] = None,
        polarity_guard: bool = True,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.index_version = index_version or "default_v1"
        self.polarity_guard = polarity_guard
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(
        self,
        query: str,
        query_embedding: Any,
        llm_model: str,
        top_k: int,
        enable_expansion: bool,
        enable_reranking: bool,
    ) -> Optional[tuple[RAGResponse, float]]:
        """
        Search for a semantically similar cached response.
        Returns (copy_of_response, similarity_score) if match found above threshold, else None.
        """
        import numpy as np

        query_vec = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        query_polarity = extract_negation_polarity(query) if self.polarity_guard else False
        now = time.time()

        with self._lock:
            best_key = None
            best_entry = None
            best_sim = -1.0

            # Linear scan over cached embeddings (max_entries <= 500 is ~0.1ms in numpy)
            for key, entry in list(self._cache.items()):
                # Parameter and version check
                if entry.get("index_version") != self.index_version:
                    continue
                if (
                    entry.get("llm_model") != llm_model
                    or entry.get("top_k") != top_k
                    or entry.get("enable_expansion") != enable_expansion
                    or entry.get("enable_reranking") != enable_reranking
                ):
                    continue

                # TTL check
                if self.ttl_seconds and (now - entry["timestamp"] > self.ttl_seconds):
                    continue

                # Polarity contradiction check
                if self.polarity_guard and (entry.get("has_negation", False) != query_polarity):
                    continue

                cached_vec = entry["embedding"]
                sim = float(np.dot(query_vec, cached_vec))
                if sim > best_sim:
                    best_sim = sim
                    best_key = key
                    best_entry = entry

            if best_entry is not None and best_sim >= self.similarity_threshold:
                self._cache.move_to_end(best_key)
                self.hits += 1
                return copy.deepcopy(best_entry["response"]), best_sim

            self.misses += 1
            return None

    def set(
        self,
        query: str,
        query_embedding: Any,
        response: RAGResponse,
        llm_model: str,
        top_k: int,
        enable_expansion: bool,
        enable_reranking: bool,
    ) -> None:
        """Store query embedding and RAGResponse in semantic cache."""
        import numpy as np

        query_vec = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        has_neg = extract_negation_polarity(query)
        now = time.time()
        key = hashlib.sha256(f"{query}|{self.index_version}|{llm_model}|{top_k}".encode("utf-8")).hexdigest()

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_entries:
                    self._cache.popitem(last=False)
                    self.evictions += 1

            self._cache[key] = {
                "query": query,
                "embedding": query_vec,
                "response": copy.deepcopy(response),
                "timestamp": now,
                "index_version": self.index_version,
                "llm_model": llm_model,
                "top_k": top_k,
                "enable_expansion": enable_expansion,
                "enable_reranking": enable_reranking,
                "has_negation": has_neg,
            }

    def update_index_version(self, new_version: str) -> None:
        """Invalidate entries on index rebuild or version change."""
        with self._lock:
            if new_version != self.index_version:
                self.index_version = new_version
                self._cache.clear()

    def clear(self) -> None:
        """Purge all semantic cache entries."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return semantic cache metrics."""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 4),
                "hit_rate_pct": round(hit_rate * 100.0, 2),
                "llm_calls_avoided": self.hits,
                "evictions": self.evictions,
                "index_version": self.index_version,
                "similarity_threshold": self.similarity_threshold,
            }
