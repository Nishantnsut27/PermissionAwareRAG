"""Sweep the Phase 8 stopping rule on the DEV split only (spec sections 28/44).

The ablation showed every coverage signal leaves the document COUNT unchanged:
the selector still fills its capacity, so precision cannot move. The stopping
rule - "is the best remaining candidate still worth a slot?" - is the only
lever that shrinks the evidence set, so it is the thing to tune.

Retrieval and the Phase 7 ranking are built once per case and reused for every
stop value, so this is a pure in-memory sweep with no API calls.

Guard: reject any value that costs recall against the frozen Phase 7 baseline.
"""
from __future__ import annotations

import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from access_control.src import audit
from access_control.src.engine import PermissionEngine
from access_control.src.retrieval import MAX_CANDIDATES
from access_control.src.users import get_user
from answering.src import config, query_aware_rerank, selection
from answering.src.evidence_selection import EvidenceWeights
from evaluation import (dataset as ds, metrics, phase8_split,
                        run_stage_eval as rse)
from evaluation.run_final_metrics_eval import build_pool, plan_from_record
from evaluation.run_phase8_eval import (aspect_coverage, load_cache,
                                        redundancy_rate)

ENGINE = PermissionEngine()
STOPS = [0.0, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]


def build(case_ids: set[str]):
    """Phase 7 ranking per case, computed once."""
    dataset = ds.load()
    cache = load_cache()
    saved = rse.load_saved_plans(
        ds.PACKAGE_ROOT / "runs" / "query-understanding-eval.json")
    cfg = config.DEFAULT_ANSWERING
    built = []
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
        fused = rse.phase3_retrieve(
            text, top_k=min(max(cfg.candidate_pool * 4, 10), MAX_CANDIDATES),
            filters=scope)
        pool = build_pool(fused, plan, user, f"sweep-{case.eval_id}",
                          rework=True)
        raw = cache[case.eval_id]["plain"]
        surviving = [(c, raw[c.chunk_id]) for c in pool
                     if c.chunk_id in raw
                     and raw[c.chunk_id] >= cfg.rerank_threshold]
        items = [(c, query_aware_rerank.adjust(s, c, plan))
                 for c, s in surviving]
        items.sort(key=lambda pair: pair[1], reverse=True)
        if not items:
            continue
        by_document: dict[str, str] = {}
        for chunk, _ in items:
            by_document.setdefault(chunk.document_id, "")
            by_document[chunk.document_id] += " " + (chunk.text or "")
        built.append((case, plan, items, by_document,
                      list(dict.fromkeys(case.expected_sources))))
        print(f"  built {case.eval_id}", flush=True)
    return built, cfg


def evaluate(built, cfg, weights, plan_aware: bool):
    rows = []
    for case, plan, items, by_document, expected in built:
        chosen, _stats = selection.select(
            items, cfg, plan=plan if plan_aware else None,
            weights=weights if plan_aware else None)
        documents = list(dict.fromkeys(
            c.document_id for c, _ in chosen if c.document_id))
        scored = metrics.score_retrieval(expected, documents)
        rows.append({
            "eval_id": case.eval_id,
            "recall": scored.recall_at_k, "precision": scored.precision_at_k,
            "mrr": scored.mrr, "n_documents": len(documents),
            "n_chunks": len(chosen),
            "aspect_coverage": aspect_coverage(plan, documents, by_document),
            "redundancy": redundancy_rate(documents, by_document),
        })
    return rows


def mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.mean(values), 4) if values else None


def main() -> int:
    config.validate_env()
    audit.AUDIT_FILE = ds.PACKAGE_ROOT / "runs" / "phase8-sweep-audit.jsonl"
    dev, _holdout = phase8_split.split()
    print(f"building Phase 7 rankings for {len(dev)} dev cases...")
    built, cfg = build(set(dev))
    print(f"built {len(built)} cases\n")

    base = EvidenceWeights.from_config()
    baseline = evaluate(built, cfg, None, plan_aware=False)
    b = {k: mean(baseline, k) for k in
         ("recall", "precision", "mrr", "n_documents", "n_chunks",
          "aspect_coverage", "redundancy")}
    print(f"{'stop':>6} {'recall':>8} {'prec':>8} {'mrr':>8} {'docs':>7} "
          f"{'chunks':>7} {'aspect':>7} {'redund':>7}  {'recall regressions'}")
    print(f"{'P7':>6} {b['recall']:>8.4f} {b['precision']:>8.4f} "
          f"{b['mrr']:>8.4f} {b['n_documents']:>7.2f} {b['n_chunks']:>7.2f} "
          f"{b['aspect_coverage']:>7.3f} {b['redundancy']:>7.4f}  -")

    by_case_baseline = {r["eval_id"]: r for r in baseline}
    for stop in STOPS:
        rows = evaluate(built, cfg, replace(base, min_marginal_utility=stop),
                        plan_aware=True)
        m = {k: mean(rows, k) for k in
             ("recall", "precision", "mrr", "n_documents", "n_chunks",
              "aspect_coverage", "redundancy")}
        regressions = [r["eval_id"] for r in rows
                       if (r["recall"] or 0)
                       < (by_case_baseline[r["eval_id"]]["recall"] or 0) - 1e-9]
        flag = "" if not regressions else " ".join(regressions)
        print(f"{stop:>6.2f} {m['recall']:>8.4f} {m['precision']:>8.4f} "
              f"{m['mrr']:>8.4f} {m['n_documents']:>7.2f} {m['n_chunks']:>7.2f} "
              f"{m['aspect_coverage']:>7.3f} {m['redundancy']:>7.4f}  {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
