"""
Phase 4 Chunking Strategies Benchmark:
Compares:
  1. recursive_small (chunk_size=500, overlap=100, strategy="recursive")
  2. recursive_medium (chunk_size=1000, overlap=200, strategy="recursive" — baseline)
  3. recursive_large (chunk_size=1500, overlap=300, strategy="recursive")
  4. structure_aware (chunk_size=750, overlap=150, strategy="structure_aware")

Measures:
  - Total chunks generated from corpus
  - Chunking & embedding latency
  - Average chunk length (characters)
  - Query retrieval recall and precision
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
RESULTS_DIR = Path(__file__).parent / "results"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Target queries for fixture documents (doc_alpha.txt and doc_beta.txt)
EVAL_QUERIES = [
    {
        "query": "what is gradient descent learning rate",
        "relevant_terms": ["gradient descent", "learning rate", "loss", "weights"],
    },
    {
        "query": "how does the attention mechanism work",
        "relevant_terms": ["attention", "transformer", "query", "key", "value"],
    },
    {
        "query": "what is backpropagation",
        "relevant_terms": ["backpropagation", "gradients", "derivatives", "chain rule"],
    },
]


def run_chunking_benchmark():
    print("=" * 65)
    print("Phase 4: Document Chunking Strategy Benchmark")
    print("=" * 65)

    # Load fixture documents
    docs = []
    for txt_file in sorted(FIXTURE_DIR.glob("*.txt")):
        docs.append(
            Document(
                page_content=txt_file.read_text(encoding="utf-8"),
                metadata={"source": txt_file.name},
            )
        )
    print(f"[INFO] Loaded {len(docs)} fixture documents.")

    configs = [
        {"name": "recursive_small", "chunk_size": 300, "overlap": 50, "strategy": "recursive"},
        {"name": "recursive_medium", "chunk_size": 600, "overlap": 100, "strategy": "recursive"},
        {"name": "recursive_large", "chunk_size": 1000, "overlap": 200, "strategy": "recursive"},
        {"name": "structure_aware", "chunk_size": 500, "overlap": 100, "strategy": "structure_aware"},
    ]

    benchmark_results = {}

    for cfg in configs:
        name = cfg["name"]
        c_size = cfg["chunk_size"]
        c_overlap = cfg["overlap"]
        strat = cfg["strategy"]

        start_time = time.perf_counter()

        pipeline = EmbeddingPipeline(
            model_name=EMBEDDING_MODEL,
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            strategy=strat,
        )
        chunks = pipeline.chunk_documents(docs)
        chunk_lengths = [len(c.page_content) for c in chunks] if chunks else [0]
        avg_chunk_len = sum(chunk_lengths) / len(chunk_lengths) if chunks else 0

        # Build FAISS in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FaissVectorStore(
                persist_dir=tmp_dir,
                embedding_model=EMBEDDING_MODEL,
                chunk_size=c_size,
                chunk_overlap=c_overlap,
                chunking_strategy=strat,
            )
            store.build_from_documents(docs)
            build_time_s = time.perf_counter() - start_time

            # Query retrieval metrics
            recalls = []
            precisions = []

            for q_obj in EVAL_QUERIES:
                q = q_obj["query"]
                terms = q_obj["relevant_terms"]
                results = store.query(q, top_k=3)
                retrieved_texts = [
                    (r.get("metadata") or {}).get("text", "").lower()
                    for r in results
                ]

                has_rel = any(any(term in t for term in terms) for t in retrieved_texts)
                recalls.append(1.0 if has_rel else 0.0)

                rel_count = sum(1 for t in retrieved_texts if any(term in t for term in terms))
                precisions.append(rel_count / len(retrieved_texts) if retrieved_texts else 0.0)

        benchmark_results[name] = {
            "chunk_size": c_size,
            "chunk_overlap": c_overlap,
            "strategy": strat,
            "total_chunks": len(chunks),
            "avg_chunk_length_chars": round(avg_chunk_len, 1),
            "build_and_embed_time_s": round(build_time_s, 2),
            "recall_at_3": round(sum(recalls) / len(recalls), 3),
            "precision_at_3": round(sum(precisions) / len(precisions), 3),
        }

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Config Name':<18} | {'Strategy':<15} | {'Chunks':<6} | {'AvgLen':<6} | {'Recall':<6} | {'P@3':<5}")
    print("-" * 70)
    for name, r in benchmark_results.items():
        print(
            f"{name:<18} | {r['strategy']:<15} | {r['total_chunks']:<6} | "
            f"{r['avg_chunk_length_chars']:<6.0f} | {r['recall_at_3']:<6.3f} | {r['precision_at_3']:<5.3f}"
        )
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "chunking_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"[INFO] Chunking benchmark saved to {out_path}")
    return benchmark_results


if __name__ == "__main__":
    run_chunking_benchmark()
