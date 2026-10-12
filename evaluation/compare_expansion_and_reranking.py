"""
Phase 5 Evaluation: Query Expansion and Cross-Encoder Reranking Benchmark.
Compares:
  1. baseline_rrf: Standard hybrid RRF (no expansion, no rerank)
  2. rrf_expansion: RRF with bounded query expansion
  3. rrf_rerank: RRF candidate pool re-scored by Cross-Encoder
  4. rrf_expansion_and_rerank: Both expansion and cross-encoder reranking

Measures:
  - Recall@5
  - Precision@5
  - MRR
  - Latency (mean, p50, p95 in ms)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.run_baseline import DEFAULT_DATASET, chunk_is_relevant, load_questions
from src.query_transform import expand_query_heuristic
from src.reranker import Reranker
from src.search import RAGSearch, RetrievalResult

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "comparison_phase5.json"


def run_benchmark(top_k: int = 5):
    print("=" * 65)
    print("Phase 5: Query Expansion & Cross-Encoder Reranking Benchmark")
    print("=" * 65)

    questions = load_questions(DEFAULT_DATASET)
    eval_questions = [q for q in questions if q.get("expected_behavior") != "abstain"]
    print(f"[INFO] Loaded {len(eval_questions)} answerable evaluation questions.")

    # Initialize base RAG search
    search = RAGSearch(
        persist_dir="faiss_store",
        embedding_model="BAAI/bge-base-en-v1.5",
        enable_expansion=False,
        enable_reranking=False,
    )

    reranker = Reranker(
        model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        enabled=True,
    )

    modes = [
        "baseline_rrf",
        "rrf_expansion",
        "rrf_rerank",
        "rrf_expansion_and_rerank",
    ]
    metrics_by_mode = {}

    for mode in modes:
        recalls = []
        precisions = []
        reciprocal_ranks = []
        latencies_ms = []

        print(f"\n[BENCHMARK] Evaluating mode: {mode}...")

        for q in eval_questions:
            q_text = q["question"]
            fragments = q.get("relevant_text_fragments", [])

            start_t = time.perf_counter()

            # 1. Query expansion step
            if mode in ("rrf_expansion", "rrf_expansion_and_rerank"):
                query_variants = expand_query_heuristic(q_text, max_queries=2)
            else:
                query_variants = [q_text]

            # 2. Candidate retrieval (dense + sparse) across query variants
            all_faiss = []
            all_bm25 = []
            candidate_k = 15 if mode in ("rrf_rerank", "rrf_expansion_and_rerank") else top_k

            for q_var in query_variants:
                raw_dense = search.vectorstore.query(q_var, top_k=candidate_k)
                all_faiss.extend(raw_dense)

                if search.bm25 is not None:
                    search.bm25.k = candidate_k
                    raw_sparse = search.bm25.invoke(q_var)
                    all_bm25.extend(raw_sparse)

            # 3. RRF Fusion
            fused_candidates: list[RetrievalResult] = search._reciprocal_rank_fusion(
                faiss_results=all_faiss,
                bm25_docs=all_bm25,
                top_k=candidate_k,
                return_structured=True,
            )

            # 4. Cross-Encoder Reranking
            if mode in ("rrf_rerank", "rrf_expansion_and_rerank"):
                final_results = reranker.rerank(
                    query=q_text,
                    candidates=fused_candidates,
                    top_k=top_k,
                )
            else:
                final_results = fused_candidates[:top_k]

            duration_ms = (time.perf_counter() - start_t) * 1000.0
            latencies_ms.append(duration_ms)

            retrieved_texts = [res.text for res in final_results]

            # Retrieval accuracy metrics
            found_any = any(chunk_is_relevant(t, fragments) for t in retrieved_texts)
            recalls.append(1.0 if found_any else 0.0)

            rel_count = sum(1 for t in retrieved_texts if chunk_is_relevant(t, fragments))
            precisions.append(rel_count / len(retrieved_texts) if retrieved_texts else 0.0)

            # MRR
            rr = 0.0
            for r_idx, t in enumerate(retrieved_texts, start=1):
                if chunk_is_relevant(t, fragments):
                    rr = 1.0 / r_idx
                    break
            reciprocal_ranks.append(rr)

        latencies_ms.sort()
        p50 = latencies_ms[len(latencies_ms) // 2]
        p95 = latencies_ms[int(len(latencies_ms) * 0.95)]

        metrics_by_mode[mode] = {
            "recall_at_5": round(sum(recalls) / len(recalls), 4),
            "precision_at_5": round(sum(precisions) / len(precisions), 4),
            "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
            "latency_mean_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
        }

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Retrieval Configuration':<28} | {'Recall@5':<9} | {'P@5':<7} | {'MRR':<7} | {'Latency Mean':<12} | {'P95 (ms)':<8}")
    print("-" * 80)
    for mode, m in metrics_by_mode.items():
        print(
            f"{mode:<28} | {m['recall_at_5']:<9.3f} | {m['precision_at_5']:<7.3f} | "
            f"{m['mrr']:<7.3f} | {m['latency_mean_ms']:<10.2f}ms | {m['latency_p95_ms']:<8.2f}"
        )
    print("=" * 80)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_by_mode, f, indent=2)
    print(f"\n[INFO] Benchmark saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_benchmark(top_k=5)
