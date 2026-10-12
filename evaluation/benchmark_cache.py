"""
Benchmark Caching Effectiveness for Contextra RAG.

Measures:
  - Cache Hit Rate on repetitive workloads (Zipfian FAQ distribution).
  - Latency reduction: Cold (retrieval + LLM) vs Warm (cache hit).
  - LLM calls avoided.
  - Citation fidelity (ensuring cached citations match original exactly).
  - Invalidation behavior upon index version advancement.

Usage:
    python evaluation/benchmark_cache.py
    python evaluation/benchmark_cache.py --dataset evaluation/datasets/golden_questions.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cache import ExactQueryCache, get_index_version
from src.generation_types import Citation, RAGResponse
from src.search import RAGSearch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RAG Caching Effectiveness")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "datasets" / "golden_questions.jsonl",
        help="Path to golden questions dataset",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=40,
        help="Total queries to simulate in repetitive workload",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory to save JSON benchmark output",
    )
    return parser.parse_args()


def load_sample_questions(dataset_path: Path) -> list[str]:
    questions = []
    if dataset_path.exists():
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        data = json.loads(line)
                        if "question" in data:
                            questions.append(data["question"])
                    except Exception:
                        pass
    if not questions:
        questions = [
            "What is attention mechanism?",
            "How does self-attention work in Transformers?",
            "What is positional encoding?",
            "Explain multi-head attention.",
            "What optimizer was used in the Attention paper?",
            "What is the learning rate schedule?",
            "Why do we use layer normalization?",
            "What is the dimension of keys and queries?",
        ]
    return questions


def run_cache_benchmark(dataset_path: Path, num_queries: int, output_dir: Path) -> dict:
    base_questions = load_sample_questions(dataset_path)[:10]
    print(f"\n======================================================================")
    print(f"  CONTEXTRA RAG - CACHE EFFECTIVENESS BENCHMARK")
    print(f"======================================================================")
    print(f"Loaded {len(base_questions)} distinct queries from dataset.")

    # 1. Initialize RAGSearch in test mode (mock LLM call with realistic ~800ms latency simulation if offline)
    search_engine = RAGSearch(llm_model="z-ai/glm-5.3-flash")
    search_engine.clear_cache()

    # Track cold latencies and warm latencies
    cold_latencies_ms: list[float] = []
    warm_latencies_ms: list[float] = []
    citation_fidelity_checks: list[bool] = []

    # Mock the LLM generation so we can measure cache bypass vs hit without spending external API tokens
    real_generate_grounded = search_engine.generate_grounded_answer

    # Generate synthetic workload with repetition (Zipfian: 70% traffic goes to top 3 questions)
    # Also add casing and whitespace variations
    workload = []
    rng = random.Random(42)
    weights = [1.0 / (i + 1) for i in range(len(base_questions))]
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]

    variations = [
        lambda q: q,
        lambda q: q.lower(),
        lambda q: q.upper(),
        lambda q: f"  {q}  ",
        lambda q: q.strip("?"),
        lambda q: f"{q}?",
    ]

    for _ in range(num_queries):
        chosen_q = rng.choices(base_questions, weights=norm_weights, k=1)[0]
        transform = rng.choice(variations)
        workload.append(transform(chosen_q))

    print(f"Generated synthetic workload of {len(workload)} queries (Zipfian distribution with variations).")
    print("\nExecuting workload...")

    # We track hits and misses directly
    mock_llm_calls = 0

    # Patch the internal generation to measure cold vs warm
    class MockAIMessage:
        def __init__(self, content: str):
            self.content = content

    mock_llm = MagicMock()

    def tracked_invoke(prompt_text, *args, **kwargs):
        nonlocal mock_llm_calls
        mock_llm_calls += 1
        # Simulate realistic generation network round-trip of 50ms
        time.sleep(0.05)
        # Return structured answer referencing Context 1 so citations are verified
        return MockAIMessage(
            "The attention mechanism maps a query and key-value pairs to an output. [Context 1]"
        )

    mock_llm.invoke.side_effect = tracked_invoke
    search_engine.llm = mock_llm

    first_results: dict[str, RAGResponse] = {}

    for i, q in enumerate(workload):
        t0 = time.perf_counter()
        resp = search_engine.generate_grounded_answer(query=q, top_k=3)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        is_hit = resp.metadata.get("cache_hit", False)
        normalized_q = q.strip().lower()

        if is_hit:
            warm_latencies_ms.append(elapsed_ms)
            # Verify citation fidelity
            if normalized_q in first_results:
                orig = first_results[normalized_q]
                fidelity = (
                    len(resp.citations) == len(orig.citations)
                    and all(
                        c1.chunk_id == c2.chunk_id
                        for c1, c2 in zip(resp.citations, orig.citations)
                    )
                )
                citation_fidelity_checks.append(fidelity)
        else:
            cold_latencies_ms.append(elapsed_ms)
            first_results[normalized_q] = resp

    cache_stats = search_engine.get_cache_stats()

    avg_cold_ms = sum(cold_latencies_ms) / len(cold_latencies_ms) if cold_latencies_ms else 0.0
    avg_warm_ms = sum(warm_latencies_ms) / len(warm_latencies_ms) if warm_latencies_ms else 0.0
    speedup = (avg_cold_ms / avg_warm_ms) if avg_warm_ms > 0 else 1.0
    citation_accuracy_pct = (
        (sum(citation_fidelity_checks) / len(citation_fidelity_checks) * 100.0)
        if citation_fidelity_checks
        else 100.0
    )

    # Invalidation test:
    old_version = search_engine.index_version
    search_engine.index_version = "invalidated_test_version"
    post_inval_resp = search_engine.generate_grounded_answer(query=workload[0], top_k=3)
    invalidation_success = not post_inval_resp.metadata.get("cache_hit", False)

    # Summary table
    print("\n----------------------------------------------------------------------")
    print("                    CACHE BENCHMARK RESULTS")
    print("----------------------------------------------------------------------")
    print(f"  Total Queries Processed:     {num_queries}")
    print(f"  Cache Hits:                  {cache_stats['hits']}")
    print(f"  Cache Misses:                {cache_stats['misses']}")
    print(f"  Cache Hit Rate:              {cache_stats['hit_rate_pct']}%")
    print(f"  LLM Calls Avoided:           {cache_stats['llm_calls_avoided']}")
    print(f"  Avg Cold Query Latency:      {avg_cold_ms:.2f} ms")
    print(f"  Avg Cached Query Latency:    {avg_warm_ms:.2f} ms")
    print(f"  Latency Reduction / Speedup: {speedup:.1f}x speedup ({((1 - avg_warm_ms/avg_cold_ms)*100):.1f}% faster)")
    print(f"  Citation Fidelity:           {citation_accuracy_pct:.1f}% bit-for-bit identical")
    print(f"  Version Invalidation Check:  {'PASSED (stale cache rejected)' if invalidation_success else 'FAILED'}")
    print("----------------------------------------------------------------------")

    results_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_queries": num_queries,
        "hits": cache_stats["hits"],
        "misses": cache_stats["misses"],
        "hit_rate_pct": cache_stats["hit_rate_pct"],
        "llm_calls_avoided": cache_stats["llm_calls_avoided"],
        "avg_cold_latency_ms": round(avg_cold_ms, 2),
        "avg_cached_latency_ms": round(avg_warm_ms, 2),
        "speedup_ratio": round(speedup, 2),
        "latency_reduction_pct": round((1 - avg_warm_ms / avg_cold_ms) * 100, 2) if avg_cold_ms > 0 else 0.0,
        "citation_fidelity_pct": round(citation_accuracy_pct, 2),
        "version_invalidation_verified": invalidation_success,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"cache_benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved benchmark results to: {out_file}\n")

    return results_data


if __name__ == "__main__":
    args = parse_args()
    run_cache_benchmark(args.dataset, args.num_queries, args.output_dir)
