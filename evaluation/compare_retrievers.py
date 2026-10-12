"""
Phase 3 Retrieval Comparison Benchmark:
Compares:
  1. Dense Only (FAISS IndexFlatL2 with BAAI/bge-base-en-v1.5)
  2. Sparse Only (BM25Retriever)
  3. Hybrid (FAISS + BM25 combined via Reciprocal Rank Fusion)

Measures:
  - Recall@5
  - Precision@5
  - MRR
  - Latency (p50, p95, mean in ms)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from src.vectorstore import FaissVectorStore
from src.search import RAGSearch, RetrievalResult, reciprocal_rank_fusion
from evaluation.run_baseline import load_questions, chunk_is_relevant, DEFAULT_DATASET

RESULTS_DIR = Path(__file__).parent / "results"
PERSIST_DIR = "faiss_store"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def run_benchmark(top_k: int = 5):
    print("=" * 65)
    print("Phase 3: Hybrid Retrieval & RRF Comparison Benchmark")
    print("=" * 65)

    questions = load_questions(DEFAULT_DATASET)
    eval_questions = [q for q in questions if q.get("expected_behavior") != "abstain"]
    print(f"[INFO] Loaded {len(eval_questions)} answerable evaluation questions.")

    # 1. Initialize store & BM25
    store = FaissVectorStore(persist_dir=PERSIST_DIR, embedding_model=EMBEDDING_MODEL)
    store.load()

    bm25_docs = [
        Document(
            page_content=item.get("text", ""),
            metadata={k: v for k, v in item.items() if k != "text"},
        )
        for item in store.metadata
        if item and item.get("text", "").strip()
    ]
    bm25 = BM25Retriever.from_documents(bm25_docs)
    bm25.k = top_k

    modes = ["dense_only", "bm25_only", "rrf_hybrid"]
    metrics_by_mode = {}

    for mode in modes:
        recalls = []
        precisions = []
        reciprocal_ranks = []
        latencies_ms = []

        for q in eval_questions:
            q_text = q["question"]
            fragments = q.get("relevant_text_fragments", [])

            start_t = time.perf_counter()

            if mode == "dense_only":
                raw = store.query(q_text, top_k=top_k)
                retrieved_texts = [
                    (r.get("metadata") or {}).get("text", "")
                    for r in raw
                    if r.get("metadata")
                ]
            elif mode == "bm25_only":
                raw_docs = bm25.invoke(q_text)
                retrieved_texts = [d.page_content for d in raw_docs[:top_k]]
            elif mode == "rrf_hybrid":
                # Dense candidates
                dense_raw = store.query(q_text, top_k=10)
                dense_results = [
                    RetrievalResult(
                        chunk_id=r.get("metadata", {}).get("chunk_id", f"c_{i}"),
                        text=(r.get("metadata") or {}).get("text", ""),
                        metadata=r.get("metadata") or {},
                        retriever="faiss",
                        rank=i + 1,
                        score=float(r.get("distance", 0.0)),
                    )
                    for i, r in enumerate(dense_raw)
                    if (r.get("metadata") or {}).get("text")
                ]

                # BM25 candidates
                bm25_docs_raw = bm25.invoke(q_text)
                sparse_results = [
                    RetrievalResult(
                        chunk_id=getattr(d, "metadata", {}).get("chunk_id", f"bm25_{j}"),
                        text=d.page_content,
                        metadata=getattr(d, "metadata", {}),
                        retriever="bm25",
                        rank=j + 1,
                        score=1.0 / (j + 1),
                    )
                    for j, d in enumerate(bm25_docs_raw)
                ]

                fused = reciprocal_rank_fusion([dense_results, sparse_results], rrf_k=60, top_k=top_k)
                retrieved_texts = [item["text"] for item in fused]

            duration_ms = (time.perf_counter() - start_t) * 1000.0
            latencies_ms.append(duration_ms)

            # Compute retrieval metrics
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
            "recall_at_5": sum(recalls) / len(recalls),
            "precision_at_5": sum(precisions) / len(precisions),
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "latency_mean_ms": sum(latencies_ms) / len(latencies_ms),
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
        }

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Retriever Mode':<15} | {'Recall@5':<9} | {'P@5':<7} | {'MRR':<7} | {'Latency (mean)':<14}")
    print("-" * 65)
    for mode, m in metrics_by_mode.items():
        print(
            f"{mode:<15} | {m['recall_at_5']:<9.3f} | {m['precision_at_5']:<7.3f} | "
            f"{m['mrr']:<7.3f} | {m['latency_mean_ms']:<10.2f} ms"
        )
    print("=" * 65)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "comparison_phase3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_by_mode, f, indent=2)
    print(f"[INFO] Benchmark results saved to {out_path}")
    return metrics_by_mode


if __name__ == "__main__":
    run_benchmark(top_k=5)
