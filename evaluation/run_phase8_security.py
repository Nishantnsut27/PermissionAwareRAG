"""Phase 8 security verification (spec sections 24-26).

The structural guarantees are pinned by unit tests: the selector is subtractive,
it only ever sees post-authorization chunks, and `service._prepare` runs the
final `can_access` gate after selection. This script is the empirical
belt-and-suspenders check the spec asks for: it runs the shipped Phase 8
selector over every scored golden case and RE-CHECKS every chunk it selected
against the real `PermissionEngine`, for the identity that actually asked.

It reuses the exact build path of `run_phase8_eval` (live retrieval + the frozen
Phase 7 ranking rebuilt from the Jina cache), so the pool it selects from is the
same authorized pool production would see. Zero external calls beyond the same
retrieval the evaluation already makes; no Jina, OpenJEV or LLM call is added.

A violation here would mean the selector emitted a chunk the engine denies -
i.e. Phase 8 widened access. The expected result is zero.

Usage:
    python -m evaluation.run_phase8_security
    python -m evaluation.run_phase8_security --split holdout
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from access_control.src import audit
from access_control.src.engine import PermissionEngine
from access_control.src.retrieval import MAX_CANDIDATES
from access_control.src.users import get_user
from answering.src import config, query_aware_rerank, selection
from evaluation import dataset as ds, phase8_split, run_stage_eval as rse
from evaluation.run_phase8_eval import load_cache
from evaluation.run_final_metrics_eval import build_pool, plan_from_record

ENGINE = PermissionEngine()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(case_ids: set[str]) -> dict:
    dataset = ds.load()
    cache = load_cache()
    saved = rse.load_saved_plans(
        ds.PACKAGE_ROOT / "runs" / "query-understanding-eval.json")
    cfg = config.DEFAULT_ANSWERING

    cases_checked = 0
    chunks_rechecked = 0
    documents_rechecked = 0
    violations: list[dict] = []
    forbidden_selected: list[dict] = []
    rows: list[dict] = []

    for case in dataset.cases:
        if case.eval_id not in case_ids or case.expected_behavior == "refuse":
            continue
        record = saved.get(case.eval_id)
        if record is None or case.eval_id not in cache:
            continue
        plan = plan_from_record(record, case.question)
        user = get_user(case.user_id)
        scope = ENGINE.get_authorized_scope(user, None)
        text = rse.search_text_for(case, dataset)
        top_k = min(max(cfg.candidate_pool * 4, 10), MAX_CANDIDATES)
        fused = rse.phase3_retrieve(text, top_k=top_k, filters=scope)
        pool = build_pool(fused, plan, user, f"sec-{case.eval_id}", rework=True)

        raw = cache[case.eval_id]["plain"]
        surviving = [(c, raw[c.chunk_id]) for c in pool
                     if c.chunk_id in raw
                     and raw[c.chunk_id] >= cfg.rerank_threshold]
        items = [(c, query_aware_rerank.adjust(s, c, plan))
                 for c, s in surviving]
        items.sort(key=lambda pair: pair[1], reverse=True)
        if not items:
            continue

        # Shipped Phase 8 selection: plan supplied, default (config) weights.
        chosen, _stats = selection.select(items, cfg, plan=plan)

        cases_checked += 1
        selected_docs: list[str] = []
        for chunk, _score in chosen:
            chunks_rechecked += 1
            decision = ENGINE.can_access(user, getattr(chunk, "metadata", None))
            if not decision.allow:
                violations.append({
                    "eval_id": case.eval_id, "user_id": case.user_id,
                    "chunk_id": getattr(chunk, "chunk_id", None),
                    "document_id": getattr(chunk, "document_id", None),
                    "reason": decision.reason})
            document_id = getattr(chunk, "document_id", None)
            if document_id and document_id not in selected_docs:
                selected_docs.append(document_id)
        documents_rechecked += len(selected_docs)

        forbidden = sorted(set(selected_docs) & set(case.forbidden_sources))
        if forbidden:
            forbidden_selected.append({
                "eval_id": case.eval_id, "user_id": case.user_id,
                "documents": forbidden})

        rows.append({
            "eval_id": case.eval_id, "category": case.category,
            "user_id": case.user_id,
            "selected_documents": selected_docs,
            "selected_chunks": len(chosen),
            "forbidden_intersection": forbidden,
        })
        print(f"{case.eval_id}: {len(chosen)} chunks / {len(selected_docs)} docs"
              f" rechecked, violations={sum(1 for v in violations if v['eval_id']==case.eval_id)}",
              flush=True)

    rate = (len(violations) + len(forbidden_selected)) / cases_checked \
        if cases_checked else 0.0
    return {
        "schema_version": 1,
        "generated_at": _now(),
        "cases": cases_checked,
        "chunks_rechecked": chunks_rechecked,
        "documents_rechecked": documents_rechecked,
        "permission_engine_violations": len(violations),
        "forbidden_sources_selected": len(forbidden_selected),
        "unauthorized_retrieval_rate": round(rate, 4),
        "violations": violations,
        "forbidden_details": forbidden_selected,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "holdout", "all"),
                        default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config.validate_env()
    audit.AUDIT_FILE = ds.PACKAGE_ROOT / "runs" / "phase8-security-audit.jsonl"
    dev, holdout = phase8_split.split()
    case_ids = {"dev": set(dev), "holdout": set(holdout),
                "all": set(dev) | set(holdout)}[args.split]

    document = run(case_ids)
    output = args.output or (ds.PACKAGE_ROOT / "runs"
                             / f"phase8-security-{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                      encoding="utf-8")

    print("\n=== Phase 8 security verification ===")
    for key in ("cases", "chunks_rechecked", "documents_rechecked",
                "permission_engine_violations", "forbidden_sources_selected",
                "unauthorized_retrieval_rate"):
        print(f"  {key:32}: {document[key]}")
    print(f"Results: {output.resolve()}")
    return 1 if (document["permission_engine_violations"]
                 or document["forbidden_sources_selected"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
