"""Phase 9 security verification (spec sections 28, K).

Two invariants, checked empirically over every scored golden case:

1. Subset: the chunks Phase 9 places in the context are a subset of the chunks
   Phase 8 selected (Phase 9 can drop and reorder, never introduce). This is the
   structural reason Phase 9 cannot widen access.
2. Authorization: every chunk Phase 9 places still passes the PermissionEngine
   for the identity that asked - the same gate `service._prepare` applies before
   packing. Expected result: zero violations.

Reuses the exact offline build path of run_phase9_eval (live retrieval + the
frozen Jina cache), so it adds no external call beyond that retrieval.

Usage:
    python -m evaluation.run_phase9_security
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
from access_control.src.users import get_user
from answering.src import config, context_packing
from evaluation import dataset as ds, phase8_split, run_stage_eval as rse
from evaluation.run_phase8_eval import load_cache
from evaluation.run_phase9_eval import reconstruct

ENGINE = PermissionEngine()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "holdout", "all"),
                        default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config.validate_env()
    audit.AUDIT_FILE = ds.PACKAGE_ROOT / "runs" / "phase9-security-audit.jsonl"
    dataset = ds.load()
    cache = load_cache()
    saved = rse.load_saved_plans(
        ds.PACKAGE_ROOT / "runs" / "query-understanding-eval.json")
    cfg = config.DEFAULT_ANSWERING
    dev, holdout = phase8_split.split()
    case_ids = {"dev": set(dev), "holdout": set(holdout),
                "all": set(dev) | set(holdout)}[args.split]

    cases = 0
    placed_chunks = 0
    subset_violations: list[dict] = []
    auth_violations: list[dict] = []
    forbidden_selected: list[dict] = []

    for case in dataset.cases:
        if case.eval_id not in case_ids or case.expected_behavior == "refuse":
            continue
        built = reconstruct(case, dataset, cache, saved, cfg)
        if built is None:
            continue
        plan, selected_pairs = built
        selected_ids = {getattr(c, "chunk_id", None) for c, _ in selected_pairs}
        user = get_user(case.user_id)

        _text, sources, _stats = context_packing.build_packed_with_stats(
            selected_pairs, cfg, plan)
        by_id = {getattr(c, "chunk_id", None): c for c, _ in selected_pairs}

        cases += 1
        placed_docs: list[str] = []
        for source in sources:
            for cid in source.chunk_ids:
                placed_chunks += 1
                if cid not in selected_ids:
                    subset_violations.append({"eval_id": case.eval_id,
                                              "chunk_id": cid})
                    continue
                chunk = by_id.get(cid)
                decision = ENGINE.can_access(user, getattr(chunk, "metadata", None))
                if not decision.allow:
                    auth_violations.append({"eval_id": case.eval_id,
                                            "chunk_id": cid,
                                            "reason": decision.reason})
            if source.document_id not in placed_docs:
                placed_docs.append(source.document_id)
        forbidden = sorted(set(placed_docs) & set(case.forbidden_sources))
        if forbidden:
            forbidden_selected.append({"eval_id": case.eval_id,
                                       "documents": forbidden})
        print(f"{case.eval_id}: placed {sum(len(s.chunk_ids) for s in sources)} "
              f"chunks, subset_ok={not subset_violations}, "
              f"auth_violations={len([v for v in auth_violations if v['eval_id']==case.eval_id])}",
              flush=True)

    document = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": cases,
        "placed_chunks_rechecked": placed_chunks,
        "subset_violations": len(subset_violations),
        "permission_engine_violations": len(auth_violations),
        "forbidden_sources_selected": len(forbidden_selected),
        "unauthorized_retrieval_rate": round(
            (len(auth_violations) + len(forbidden_selected)) / cases, 4)
        if cases else 0.0,
        "violation_details": {"subset": subset_violations,
                              "authorization": auth_violations,
                              "forbidden": forbidden_selected},
    }
    output = args.output or (ds.PACKAGE_ROOT / "runs"
                             / f"phase9-security-{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print("\n=== Phase 9 security verification ===")
    for key in ("cases", "placed_chunks_rechecked", "subset_violations",
                "permission_engine_violations", "forbidden_sources_selected",
                "unauthorized_retrieval_rate"):
        print(f"  {key:32}: {document[key]}")
    print(f"Results: {output.resolve()}")
    return 1 if (subset_violations or auth_violations
                 or forbidden_selected) else 0


if __name__ == "__main__":
    raise SystemExit(main())
