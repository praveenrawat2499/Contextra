"""
Phase 6 Evaluation: Faithfulness, Citations, and Guardrails Benchmark.
Evaluates:
  1. Abstention accuracy on out-of-domain / unanswerable / prompt-injection queries.
  2. Citation validity rate and hallucination prevention.
  3. Grounding rate on answerable corpus questions.
  4. Prompt injection resistance (no secret leak, no rule override).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.run_baseline import DEFAULT_DATASET, load_questions
from src.search import RAGSearch
from src.guardrails import SECRET_PATTERNS

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "faithfulness_phase6.json"


def evaluate_faithfulness():
    print("=" * 65)
    print("Phase 6: Faithfulness, Citations, & Guardrails Evaluation")
    print("=" * 65)

    questions = load_questions(DEFAULT_DATASET)
    print(f"[INFO] Loaded {len(questions)} evaluation questions.")

    search = RAGSearch(
        persist_dir="faiss_store",
        embedding_model="BAAI/bge-base-en-v1.5",
        enable_expansion=False,
        enable_reranking=True,
    )

    answerable_queries = [q for q in questions if q.get("expected_behavior") != "abstain"]
    abstain_queries = [q for q in questions if q.get("expected_behavior") == "abstain"]

    print(f"[INFO] Running evaluation on {len(answerable_queries)} answerable questions...")
    answerable_results = []
    total_citations_generated = 0
    total_citations_verified = 0
    total_citations_unverified = 0

    for q in answerable_queries:
        resp = search.generate_grounded_answer(query=q["question"], top_k=5)
        total_citations_generated += len(resp.citations) + len(resp.unverified_citations)
        total_citations_verified += len(resp.citations)
        total_citations_unverified += len(resp.unverified_citations)

        answerable_results.append({
            "id": q["id"],
            "question": q["question"],
            "status": resp.status,
            "citations_count": len(resp.citations),
            "unverified_citations_count": len(resp.unverified_citations),
            "answer_preview": resp.answer[:120],
        })

    print(f"[INFO] Running evaluation on {len(abstain_queries)} unanswerable/adversarial questions...")
    abstain_results = []
    correct_abstentions = 0
    secret_leaks = 0

    for q in abstain_queries:
        resp = search.generate_grounded_answer(query=q["question"], top_k=5)
        is_abstained = resp.status == "insufficient_evidence"

        # Check if any secret leak happened
        has_secret = any(p.search(resp.answer) for p in SECRET_PATTERNS)
        if has_secret:
            secret_leaks += 1

        if is_abstained:
            correct_abstentions += 1

        abstain_results.append({
            "id": q["id"],
            "question": q["question"],
            "status": resp.status,
            "is_abstained": is_abstained,
            "answer_preview": resp.answer[:120],
        })

    # Summary metrics
    citation_validity_rate = (
        total_citations_verified / total_citations_generated
        if total_citations_generated > 0
        else 1.0
    )
    abstention_accuracy = correct_abstentions / len(abstain_queries) if abstain_queries else 1.0
    grounding_rate = sum(1 for r in answerable_results if r["status"] == "answered") / len(answerable_results)

    summary = {
        "total_questions": len(questions),
        "answerable_count": len(answerable_queries),
        "unanswerable_count": len(abstain_queries),
        "abstention_accuracy": round(abstention_accuracy, 4),
        "citation_validity_rate": round(citation_validity_rate, 4),
        "unverified_citations_allowed": total_citations_unverified,
        "grounding_rate": round(grounding_rate, 4),
        "secret_leak_count": secret_leaks,
        "answerable_details": answerable_results,
        "abstain_details": abstain_results,
    }

    print("\n" + "=" * 70)
    print("Phase 6 Evaluation Summary:")
    print(f"  Abstention Accuracy:       {abstention_accuracy * 100:.1f}% ({correct_abstentions}/{len(abstain_queries)})")
    print(f"  Citation Validity Rate:    {citation_validity_rate * 100:.1f}% ({total_citations_verified}/{total_citations_generated})")
    print(f"  Unverified Citations Leaked: {total_citations_unverified} (0 expected)")
    print(f"  Grounding Rate:            {grounding_rate * 100:.1f}%")
    print(f"  Secret Leak Count:         {secret_leaks} (0 expected)")
    print("=" * 70)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[INFO] Saved evaluation results to {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate_faithfulness()
