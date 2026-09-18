"""Pipeline entry point: index the corpus and run retrieval sanity checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.indexing import run_indexing  # noqa: E402
from src.retrieval import retrieve  # noqa: E402


def run_sanity(top_k: int = 5) -> list[dict]:
    outcomes: list[dict] = []
    for label, query in config.SANITY_QUERIES:
        try:
            results = retrieve(query, top_k=top_k)
            outcomes.append({
                "label": label,
                "query": query,
                "status": "ok" if results else "no_results",
                "results": [
                    {
                        "rank": i + 1,
                        "method": r.retrieval_method,
                        "score": round(r.score, 4),
                        "chunk_id": r.chunk_id,
                        "document_id": r.document_id,
                        "seller_id": r.metadata.get("seller_id"),
                        "department": r.metadata.get("department"),
                        "classification": r.metadata.get("classification"),
                        "document_type": r.metadata.get("document_type"),
                        "source_file": r.metadata.get("source_file"),
                        "page_numbers": r.metadata.get("page_numbers"),
                        "snippet": r.text[:200].replace("\n", " "),
                    }
                    for i, r in enumerate(results)
                ],
            })
        except Exception as exc:
            outcomes.append({"label": label, "query": query,
                             "status": f"error: {exc}", "results": []})
    with open(config.SANITY_REPORT_FILE, "w", encoding="utf-8") as fh:
        json.dump(outcomes, fh, indent=2)
    return outcomes


def print_sanity(outcomes: list[dict]) -> None:
    for case in outcomes:
        print(f"\n[{case['label']}] {case['query'][:80]} -> {case['status']}")
        for r in case["results"][:5]:
            print(f"  {r['rank']}. ({r['method']}, {r['score']}) "
                  f"{r['chunk_id']} doc={r['document_id']} "
                  f"seller={r['seller_id']} dept={r['department']} "
                  f"class={r['classification']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 indexing + retrieval")
    parser.add_argument("--skip-index", action="store_true",
                        help="skip indexing, only run sanity queries")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.skip_index:
        stats = run_indexing()
        print("Indexing complete")
        for key in ("chunks_received", "chunks_embedded", "embedding_failures",
                    "qdrant_points_indexed", "qdrant_points_failed",
                    "collection_name", "embedding_model",
                    "vector_dimensionality", "elapsed_s"):
            print(f"  {key}: {stats[key]}")
    outcomes = run_sanity(top_k=args.top_k)
    print_sanity(outcomes)
    print(f"\nReports: {config.INDEXING_REPORT_FILE}, {config.SANITY_REPORT_FILE}")
    failed = [c for c in outcomes if c["status"] != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
