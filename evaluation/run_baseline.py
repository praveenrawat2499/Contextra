"""
Phase 0 baseline retrieval evaluation harness.

Loads the existing persisted FAISS index and evaluates retrieval quality
against the golden question set.  No LLM or API calls are made.

Metrics computed (retrieval-only):
  - Recall@K    : fraction of "answerable" questions where retrieved
                  chunks contain at least one relevant text fragment.
  - Precision@K : fraction of retrieved chunks that contain a relevant
                  text fragment (averaged across questions).
  - MRR         : mean reciprocal rank of the first relevant chunk.

Limitation (Phase 0):
  Relevance is determined by text-fragment substring matching because the
  current system stores no stable chunk IDs in metadata.  Phase 2 will
  introduce chunk IDs enabling exact recall measurement.

Usage:
    python evaluation/run_baseline.py
    python evaluation/run_baseline.py --top-k 5 --persist-dir faiss_store
    python evaluation/run_baseline.py --dataset evaluation/datasets/golden_questions.jsonl

Output:
    evaluation/results/baseline_YYYYMMDD_HHMMSS.json
    Console summary table
"""

from __future__ import annotations

import argparse
import json
import sys
import datetime
from pathlib import Path

# Allow running from project root: python evaluation/run_baseline.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore import FaissVectorStore  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "golden_questions.jsonl"
DEFAULT_PERSIST_DIR = "faiss_store"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_TOP_K = 5
RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_questions(path: Path) -> list[dict]:
    """Load and validate the golden question JSONL file."""
    questions = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc

            # Validate required fields
            for field in ("id", "question", "expected_behavior"):
                if field not in q:
                    raise ValueError(
                        f"Question on line {lineno} missing required field '{field}'"
                    )

            if q["expected_behavior"] not in ("answer", "abstain"):
                raise ValueError(
                    f"Question {q['id']}: expected_behavior must be 'answer' or 'abstain', "
                    f"got {q['expected_behavior']!r}"
                )

            questions.append(q)

    if not questions:
        raise ValueError(f"No questions loaded from {path}")

    return questions


# ---------------------------------------------------------------------------
# Relevance judge (text-fragment proxy — Phase 0 only)
# ---------------------------------------------------------------------------


def chunk_is_relevant(chunk_text: str, relevant_fragments: list[str]) -> bool:
    """
    Return True if chunk_text contains at least one fragment from
    relevant_fragments (case-insensitive substring match).

    This is a proxy for relevance used only in Phase 0 because chunk IDs
    do not yet exist.  It will be replaced by exact chunk-ID matching
    in Phase 2.
    """
    if not relevant_fragments:
        return False
    lowered = chunk_text.lower()
    return any(frag.lower() in lowered for frag in relevant_fragments)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def reciprocal_rank(results: list[dict], fragments: list[str]) -> float:
    """Return 1/rank of the first relevant result, or 0 if none found."""
    for rank, r in enumerate(results, start=1):
        text = (r.get("metadata") or {}).get("text", "")
        if chunk_is_relevant(text, fragments):
            return 1.0 / rank
    return 0.0


def precision_at_k(results: list[dict], fragments: list[str]) -> float:
    """Fraction of retrieved chunks that are relevant."""
    if not results:
        return 0.0
    relevant = sum(
        1
        for r in results
        if chunk_is_relevant((r.get("metadata") or {}).get("text", ""), fragments)
    )
    return relevant / len(results)


def recall_at_k(results: list[dict], fragments: list[str]) -> float:
    """1.0 if any retrieved chunk is relevant, else 0.0 (binary recall proxy)."""
    for r in results:
        text = (r.get("metadata") or {}).get("text", "")
        if chunk_is_relevant(text, fragments):
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------


def evaluate_question(
    store: FaissVectorStore,
    question: dict,
    top_k: int,
) -> dict:
    """Run retrieval for a single question and compute metrics."""
    qid = question["id"]
    text = question["question"]
    behavior = question["expected_behavior"]
    fragments = question.get("relevant_text_fragments", [])
    tags = question.get("tags", [])

    # Skip abstain/security/unanswerable questions for retrieval metrics
    # (there are no correct chunks to retrieve for them)
    if behavior == "abstain":
        return {
            "id": qid,
            "question": text,
            "expected_behavior": behavior,
            "tags": tags,
            "skipped": True,
            "skip_reason": "abstain question — no relevant chunks defined",
        }

    if "needs-verification" in tags:
        skipped_note = "needs-verification"
    else:
        skipped_note = None

    results = store.query(text, top_k=top_k)

    rr = reciprocal_rank(results, fragments)
    prec = precision_at_k(results, fragments)
    rec = recall_at_k(results, fragments)

    retrieved_texts = [
        (r.get("metadata") or {}).get("text", "")[:120]
        for r in results
    ]

    return {
        "id": qid,
        "question": text,
        "expected_behavior": behavior,
        "tags": tags,
        "skipped": False,
        "needs_verification": skipped_note is not None,
        "relevant_fragments": fragments,
        "recall_at_k": rec,
        "precision_at_k": prec,
        "mrr_contribution": rr,
        "retrieved_snippets": retrieved_texts,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def aggregate(per_question: list[dict], top_k: int) -> dict:
    """Compute aggregate retrieval metrics over evaluated (non-skipped) questions."""
    evaluated = [q for q in per_question if not q.get("skipped")]
    verified = [q for q in evaluated if not q.get("needs_verification")]
    total = len(evaluated)
    total_verified = len(verified)

    def _mean(field: str, subset: list[dict]) -> float:
        if not subset:
            return 0.0
        return sum(q[field] for q in subset) / len(subset)

    return {
        "top_k": top_k,
        "total_questions": len(per_question),
        "evaluated_questions": total,
        "verified_questions": total_verified,
        "skipped_questions": len(per_question) - total,
        # All evaluated (including needs-verification)
        f"recall_at_{top_k}": _mean("recall_at_k", evaluated),
        f"precision_at_{top_k}": _mean("precision_at_k", evaluated),
        "mrr": _mean("mrr_contribution", evaluated),
        # Verified only (ground truth confirmed from source)
        f"recall_at_{top_k}_verified": _mean("recall_at_k", verified),
        f"precision_at_{top_k}_verified": _mean("precision_at_k", verified),
        "mrr_verified": _mean("mrr_contribution", verified),
    }


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------


def print_report(agg: dict, per_question: list[dict]) -> None:
    k = agg["top_k"]
    print("\n" + "=" * 60)
    print("Contextra Phase 0 Baseline Retrieval Evaluation")
    print("=" * 60)
    print(f"  Questions total       : {agg['total_questions']}")
    print(f"  Evaluated             : {agg['evaluated_questions']}")
    print(f"  Verified only         : {agg['verified_questions']}")
    print(f"  Skipped (abstain)     : {agg['skipped_questions']}")
    print()
    print(f"  Recall@{k} (all evaluated)  : {agg[f'recall_at_{k}']:.3f}")
    print(f"  Precision@{k} (all)         : {agg[f'precision_at_{k}']:.3f}")
    print(f"  MRR (all evaluated)         : {agg['mrr']:.3f}")
    print()
    print(f"  Recall@{k} (verified only)  : {agg[f'recall_at_{k}_verified']:.3f}")
    print(f"  Precision@{k} (verified)    : {agg[f'precision_at_{k}_verified']:.3f}")
    print(f"  MRR (verified only)         : {agg['mrr_verified']:.3f}")
    print()
    print("  Per-question results:")
    print(f"  {'ID':<14} {'R@K':>5} {'P@K':>5} {'RR':>5}  {'tags'}")
    print("  " + "-" * 55)
    for q in per_question:
        if q.get("skipped"):
            print(f"  {q['id']:<14} {'skip':>5}  {' '.join(q.get('tags', []))}")
        else:
            r = q["recall_at_k"]
            p = q["precision_at_k"]
            rr = q["mrr_contribution"]
            nv = " [unverified]" if q.get("needs_verification") else ""
            tags_str = ",".join(q.get("tags", []))
            print(f"  {q['id']:<14} {r:>5.2f} {p:>5.2f} {rr:>5.2f}  {tags_str}{nv}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0 baseline retrieval evaluation — no LLM required."
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve."
    )
    parser.add_argument(
        "--persist-dir",
        default=DEFAULT_PERSIST_DIR,
        help="Path to the persisted FAISS index directory.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name (must match the saved index).",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the golden questions JSONL file.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    persist_dir = Path(args.persist_dir)

    # --- Validate inputs ---
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    faiss_index = persist_dir / "faiss.index"
    meta_pkl = persist_dir / "metadata.pkl"
    if not faiss_index.exists() or not meta_pkl.exists():
        print(
            f"[ERROR] FAISS index not found at {persist_dir}.\n"
            "Build the index first with 'streamlit run ui.py' or 'python app.py'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load questions ---
    print(f"[INFO] Loading questions from {dataset_path}")
    questions = load_questions(dataset_path)
    print(f"[INFO] Loaded {len(questions)} questions.")

    # --- Load index ---
    print(f"[INFO] Loading FAISS index from {persist_dir} …")
    store = FaissVectorStore(
        persist_dir=str(persist_dir),
        embedding_model=args.embedding_model,
    )
    store.load()
    print(f"[INFO] Index loaded: {store.index.ntotal} vectors, {len(store.metadata)} metadata entries.")

    # --- Evaluate ---
    print(f"[INFO] Evaluating retrieval at top_k={args.top_k} …")
    per_question = [evaluate_question(store, q, args.top_k) for q in questions]

    # --- Aggregate ---
    agg = aggregate(per_question, args.top_k)

    # --- Report ---
    print_report(agg, per_question)

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"baseline_{timestamp}.json"

    output = {
        "metadata": {
            "timestamp": timestamp,
            "dataset": str(dataset_path),
            "persist_dir": str(persist_dir),
            "embedding_model": args.embedding_model,
            "top_k": args.top_k,
            "phase": "0",
            "relevance_method": "text_fragment_substring_match",
            "note": (
                "Phase 0 proxy: relevance is measured by substring match of "
                "known text fragments. Chunk IDs will be added in Phase 2 "
                "for exact recall measurement."
            ),
        },
        "aggregate": agg,
        "per_question": per_question,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Results saved to: {result_path}")


if __name__ == "__main__":
    main()
